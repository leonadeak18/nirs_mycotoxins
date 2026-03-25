import pandas as pd
import numpy as np
import time
from sklearn.model_selection import cross_val_predict, StratifiedKFold, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_recall_curve, roc_auc_score, accuracy_score, precision_score

# ==========================================
# IMPORT THE DATA
# ==========================================
from nirs.nirs_dataset import X_train, y_train, X_test, y_test
from utils.preparation import binary_transform

# Define the scopes
sources = ['intact', 'ground']
targets = ['DON', 'Total_DON']

## remove Log_Total_DON and Log_Total_DON columns from y_test
for key in y_test:
    y_test[key] = y_test[key].drop(columns=['Log_DON', 'Log_Total_DON'])
    y_train[key] = y_train[key].drop(columns=['Log_DON', 'Log_Total_DON'])

## Categorize into 2 classes based on quantiles
for source in sources:
    y_test[source], y_train[source] = binary_transform(y_test[source]), binary_transform(y_train[source])


# ==========================================
# DEFINE THE PREPROCESSING PIPELINE
# ==========================================
import chemotools.scatter as ch
from chemotools.derivative import SavitzkyGolay
preprocessing_pipelines = {
    "Raw": Pipeline([
        ('smoothing', SavitzkyGolay(window_length=15, polyorder=2, deriv = 0))
    ]),
    "SNV_Smoothing": Pipeline([
        ('snv', ch.StandardNormalVariate()),
        ('smoothing', SavitzkyGolay(window_length=15, polyorder=2, deriv = 0))
    ]),
    "EMSC_Smoothing": Pipeline([
        ('emsc', ch.ExtendedMultiplicativeScatterCorrection()),
        ('smoothing', SavitzkyGolay(window_length=15, polyorder=2, deriv = 0))
    ]),
    "1st_derivative": Pipeline([
        ('first_derivative', SavitzkyGolay(window_length=15, polyorder=2, deriv = 1))
    ]),
    "2nd_derivative": Pipeline([
        ('second_derivative', SavitzkyGolay(window_length=15, polyorder=2, deriv = 2))
    ]),
    "RNV_Smoothing": Pipeline([
        ('rnv', ch.RobustNormalVariate()),
        ('smoothing', SavitzkyGolay(window_length=15, polyorder=2, deriv = 0))
    ])
}

# ==========================================================
# DEFINE THE FEATURE SELECTION AND EXTRACTION METHODS
# ===========================================================
from sklearn.decomposition import PCA, FastICA, FactorAnalysis
from sklearn.feature_selection import SelectPercentile, SelectFdr, f_classif, SelectFromModel, RFE
from sklearn.linear_model import LogisticRegression

feature_pipelines = {
    "Full_spectra": None,
    "PCA": Pipeline([
        ('PCA', PCA(n_components=0.99, random_state=270725))
    ]),
    "ICA": Pipeline([
        ('ICA', FastICA(n_components=30, max_iter=1000, random_state=270725))
    ]),
    "Factor_Analysis": Pipeline([
        ('Factor_Analysis', FactorAnalysis(n_components=30, random_state=270725))
    ]),
    "ANOVA_Pencentile": Pipeline([
        ('ANOVA_Percentile', SelectPercentile(score_func=f_classif, percentile=20))
    ]),
    "ANOVA_FDR": Pipeline([
        ('ANOVA_FDR', SelectFdr(score_func=f_classif, alpha=0.05))
    ]),
    "Lasso L1 (Selector)": Pipeline([
        # C=0.1 applies strong regularization to drop many features
        ('Lasso_L1', SelectFromModel(LogisticRegression(penalty='l1', solver='liblinear', C=0.1, random_state=270725)))
    ]),
    
    "RFE": Pipeline([
        # step=0.1 means it drops 5% of features at a time to run faster
        ('RFE', RFE(estimator=SVC(kernel='linear', random_state=270726), n_features_to_select= 30, step=0.05))
    ])
}

# ==========================================
# SVM MODEL OPTIMIZATION
# ==========================================
experiment_results = []
# Define the grid of parameters for the SVM to test
svm_param_dist = {
    'C': [0.1, 1, 5, 10, 50, 100, 500, 1000],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
}

for source in sources:
    for target in targets:
        print(f"Evaluating: {source} samples for {target}")
        
        # Extract the specific data arrays
        X_tr = X_train[source].values
        X_te = X_test[source].values
        
        # Ensure target variables are 1D arrays
        y_tr = y_train[source][target].values
        y_te = y_test[source][target].values

        #samples weight
        tr_weights = y_train[source][target + '_weight'].values
        te_weights = y_test[source][target + '_weight'].values
        
        for pipe_name, preprocessor in preprocessing_pipelines.items():
            # Apply Scatter Corrections / Derivatives
            X_train_prep = preprocessor.fit_transform(X_tr)
            X_test_prep = preprocessor.transform(X_te)

            for fp_name, fp_method in feature_pipelines.items():
                print(f"Running -> Prep: {pipe_name} | Feature: {fp_name}...", end=" ")
                start_time = time.time()
                
                # Apply Feature Extraction / Selection
                try:
                    if fp_method is not None:
                        X_tr_final = fp_method.fit_transform(X_train_prep, y_tr)
                        X_te_final = fp_method.transform(X_test_prep)
                    else:
                        X_tr_final = X_train_prep
                        X_te_final = X_test_prep
                        
                    # Skip if a selector dropped ALL features (happens rarely with Lasso/FDR)
                    if X_tr_final.shape[1] == 0:
                        print("Skipped (0 features selected).")
                        continue

                    # --- SVM OPTIMIZATION ---
                    base_svm = SVC(kernel='rbf', class_weight='balanced', random_state=270725)
                    
                    # 1. Run Randomized Search to find the best C and Gamma
                    random_search = RandomizedSearchCV(
                        estimator=base_svm,
                        param_distributions=svm_param_dist,
                        n_iter=50,          
                        cv=10,
                        scoring='roc_auc',  
                        n_jobs=-1,          
                        random_state=270725
                    )
                    random_search.fit(X_tr_final, y_tr, sample_weight=tr_weights)
                    
                    # 2. Extract the winning SVM and Calibrate it
                    best_svm = random_search.best_estimator_
                    best_svm.probability = True # Required for calibration
                    
                    calibrated_svm = CalibratedClassifierCV(best_svm, cv=10)
                    calibrated_svm.fit(X_tr_final, y_tr, sample_weight=tr_weights)
                    
                    # 3. Test on the unseen Test Set
                    test_probs = calibrated_svm.predict_proba(X_te_final)[:, 1]
                    test_auc = roc_auc_score(y_te, test_probs, sample_weight=te_weights)
                    
                    elapsed_time = time.time() - start_time
                    print(f"Done! (Test AUC: {test_auc:.3f})")
                    
                    # 4. Save the results
                    experiment_results.append({
                        'Sample_Source': source,
                        'Target_Toxin': target,
                        'Preprocessing': pipe_name,
                        'Feature_Method': fp_name,
                        'Best_C': random_search.best_params_['C'],
                        'Best_Gamma': random_search.best_params_['gamma'],
                        'CV_Val_AUC': random_search.best_score_,
                        'Test_AUC': test_auc,
                        'Time_Sec': np.round(elapsed_time, 2)
                    })
                    
                except Exception as e:
                    print(f"Failed. Error: {str(e)}")
#combine all results into a dataframe for easier analysis
optimizied_algorithm = pd.DataFrame(experiment_results)
optimizied_algorithm.to_csv('./results/optimized_SVM_results.csv', index=False)

# ==========================================
# ROBUST THRESHOLD FINDING FOR SVM
# ==========================================
optimized_model = {
    "intact_DON": Pipeline ([
        ('1st_derivative', preprocessing_pipelines["1st_derivative"]),
        ('ANOVA_FDR', feature_pipelines["ANOVA_FDR"]),
        ('svc', SVC(kernel='rbf', C=500, gamma='scale', class_weight='balanced', random_state=270725, probability=True))
        ]),
    "intact_Total_DON": Pipeline([
        ('RNV_Smoothing', preprocessing_pipelines["RNV_Smoothing"]),
        ('Factor_Analysis', feature_pipelines["Factor_Analysis"]),
        ('svc', SVC(kernel='rbf', C=5, gamma=0.001, class_weight='balanced', random_state=270725, probability=True))
        ]),
    "ground_DON": Pipeline([
        ('1st_derivative', preprocessing_pipelines["1st_derivative"]),
        ('Factor_Analysis', feature_pipelines["Factor_Analysis"]),
        ('svc', SVC(kernel='rbf', C=0.1, gamma=0.01, class_weight='balanced', random_state=270725, probability=True))
        ]),
    "ground_Total_DON": Pipeline([
        ('RNV_Smoothing', preprocessing_pipelines["RNV_Smoothing"]),
        ('ANOVA_Pencentile', feature_pipelines["ANOVA_Pencentile"]),
        ('svc', SVC(kernel='rbf', C=500, gamma=0.01, class_weight='balanced', random_state=270725, probability=True))
        ])
}

all_results = []
production_models = {}
calibration_methods = ["sigmoid", "isotonic"]
n_repeats = 1000
target_precision = 0.95
threshold_results = {}

for source in sources:
    threshold_results[source] = {}
    for target in targets:
        threshold_results[source][target] = {}
        
        # Extract the specific data arrays for this loop iteration
        X_tr = X_train[source].values
        
        # Ensure target variables are 1D arrays
        y_tr = y_train[source][target].values

        # Convert HC to 1 and LC to 0 for precision-recall calculations
        y_tr = np.where(y_tr == 'HC', 1, 0)

        #samples weight
        tr_weights = y_train[source][target + '_weight'].values

        dataset_key = f"{source}_{target}"
        
        production_models[dataset_key] = {}

        if dataset_key not in optimized_model:
            continue
        print(f"Processing: {dataset_key}")
        base_pipeline = optimized_model[dataset_key]
        pipe_name = list(base_pipeline.named_steps.keys())[0]
        fp_name = list(base_pipeline.named_steps.keys())[1]

        for method in calibration_methods:
            threshold_results[source][target][method] = {}
            print(f"  -> Applying Calibration with {method} method...")
            calibrated_model = CalibratedClassifierCV(base_pipeline, cv=10, method=method)

            upper_thresholds = []
            lower_thresholds = []

            for i in range(n_repeats):
                cv_strategy = StratifiedKFold(n_splits=10, shuffle=True, random_state=i)
                cv_probs = cross_val_predict(calibrated_model, X_tr, y_tr, 
                    cv=cv_strategy, method='predict_proba', n_jobs=-1,
                    fit_params={'svc__sample_weight': tr_weights})
                
                # Calculate Upper Threshold (High Contamination)
                prec_high, _, thresh_high = precision_recall_curve(y_tr, cv_probs[:, 1])
                valid_idx_high = np.where(prec_high[:-1] >= target_precision)[0]
                if len(valid_idx_high) > 0:
                    upper_thresholds.append(thresh_high[valid_idx_high[0]])
                    
                # Calculate Lower Threshold (Low Contamination)
                prec_low, _, thresh_low = precision_recall_curve(1 - y_tr, cv_probs[:, 0])
                valid_idx_low = np.where(prec_low[:-1] >= target_precision)[0]
                if len(valid_idx_low) > 0:
                    lower_thresholds.append(1 - thresh_low[valid_idx_low[0]])
            
            # Save results to a dataframe for this combination
            all_results.append({
                "Material": source,
                "Target": target,
                "Preprocessing": pipe_name,
                "Feature_Selection": fp_name,
                "Calibration_Method": method,
                "Median_Upper_Thresholds": np.median(upper_thresholds),
                'Average_Upper_Thresholds': np.mean(upper_thresholds),
                'Stdev_Upper_Thresholds': np.std(upper_thresholds),
                'Median_Lower_Thresholds': np.median(lower_thresholds),
                'Average_Lower_Thresholds': np.mean(lower_thresholds),
                'Stdev_Lower_Thresholds': np.std(lower_thresholds),
                "repeats": n_repeats
            })
            threshold_results[source][target][method]['upper'] = upper_thresholds
            threshold_results[source][target][method]['lower'] = lower_thresholds

            #save the calibrated model for validation on the test set later
            production_models[dataset_key][method] = {
                "model": calibrated_model.fit(X_tr, y_tr, svc__sample_weight= tr_weights),
                "lower_threshold": np.median(lower_thresholds),
                "upper_threshold": np.median(upper_thresholds)
            }

## Export all important results
### export the model
import joblib
joblib.dump(production_models, "./model/probability_model/production_models.pkl")

from plotnine import (
    ggplot, aes, geom_histogram, geom_vline, annotate,
    scale_fill_manual, xlim, labs, theme_minimal, theme
)

def plot_threshold_distributions(lower_thresholds, upper_thresholds, optimal_lower_thresh, optimal_upper_thresh, n_repeats):
    """
    Visualizes the distribution of CV probability thresholds using plotnine.
    """
    
    # 1. Prepare the data (plotnine requires a "long" format DataFrame)
    df_lower = pd.DataFrame({'Threshold': lower_thresholds, 'Type': 'Lower (Low Contam)'})
    df_upper = pd.DataFrame({'Threshold': upper_thresholds, 'Type': 'Upper (High Contam)'})
    
    # Combine them into one clean dataset
    plot_df = pd.concat([df_lower, df_upper], ignore_index=True)
    
    # 2. Build the plotnine figure
    p = (
        ggplot(plot_df, aes(x='Threshold', fill='Type'))
        
        # Shade the "Uncertain" area first so it stays in the background
        + annotate(
            "rect", 
            xmin=optimal_lower_thresh, xmax=optimal_upper_thresh, 
            ymin=0, ymax=np.inf, # np.inf automatically fills to the top of the chart!
            fill="gray", alpha=0.2
        )
        
        # Plot the histograms
        # position="identity" ensures they overlap rather than stacking on top of each other
        + geom_histogram(bins=2, alpha=0.6, position="identity", color="white")
        
        # Add the vertical dashed lines for the medians
        + geom_vline(xintercept=optimal_lower_thresh, color='darkblue', linetype='dashed', size=1)
        + geom_vline(xintercept=optimal_upper_thresh, color='darkred', linetype='dashed', size=1)
        
        # Add text annotations right next to the lines to show the exact values
        + annotate("text", x=optimal_lower_thresh - 0.02, y=10, label=f"{optimal_lower_thresh:.3f}", color="darkblue", angle=90)
        + annotate("text", x=optimal_upper_thresh + 0.02, y=10, label=f"{optimal_upper_thresh:.3f}", color="darkred", angle=90)
        
        # Set colors and limits
        + scale_fill_manual(values={'Lower (Low Contam)': 'blue', 'Upper (High Contam)': 'red'})
        + xlim(0, 1)
        
        # Add beautiful labels
        + labs(
            title=f'Distribution of 95% Precision Thresholds over {n_repeats} CV Repeats',
            x='Probability Score',
            y='Frequency',
            fill='Threshold Type'
        )
        
        # Apply a clean, modern theme
        + theme_minimal()
        + theme(
            legend_position='top',
            figure_size=(10, 6)
        )
    )
    
    return p

my_plot = plot_threshold_distributions(lower_thresholds, upper_thresholds, np.median(lower_thresholds), np.median(upper_thresholds), n_repeats)
my_plot.show()


# ==========================================
# VERIFICATION ON TEST SET
# ==========================================
validation_summary = []

for source in sources:
    for target in targets:
        dataset_key = f"{source}_{target}"
        
        if dataset_key not in production_models:
            continue
            
        # 1. Retrieve the model and saved thresholds
        registry_entry = production_models[dataset_key]["sigmoid"]
        model = registry_entry["model"]
        lower_t = registry_entry["lower_threshold"]
        upper_t = registry_entry["upper_threshold"]
        
        # 2. Extract Test Data and map target to Binary
        X_te = X_test[source].values
        y_te_binary = np.where(y_test[source][target].values == 'HC', 1, 0)
        y_te_strings = np.where(y_te_binary == 1, 'High Contamination', 'Low Contamination')
        
        # 3. Generate Predictions
        probs = model.predict_proba(X_te)[:, 1]
        
        predictions = np.full(probs.shape, 'Uncertain', dtype=object)
        predictions[probs < lower_t] = 'Low Contamination'
        predictions[probs > upper_t] = 'High Contamination'
        
        # 4. Filter for Confident Predictions
        eval_df = pd.DataFrame({'True_Label': y_te_strings, 'Prediction': predictions})
        confident_df = eval_df[eval_df['Prediction'] != 'Uncertain']
        
        uncertain_count = len(eval_df) - len(confident_df)
        coverage_pct = (len(confident_df) / len(eval_df)) * 100
        
        # 5. Calculate Metrics
        if not confident_df.empty:
            hc_precision = precision_score(
                confident_df['True_Label'], 
                confident_df['Prediction'], 
                pos_label='High Contamination', 
                zero_division=0
            )
            accuracy = accuracy_score(confident_df['True_Label'], confident_df['Prediction'])
        else:
            hc_precision = np.nan
            accuracy = np.nan
            
        validation_summary.append({
            "Dataset": dataset_key,
            "Lower_Threshold": round(lower_t, 3),
            "Upper_Threshold": round(upper_t, 3),
            "Total_Test_Samples": len(eval_df),
            "Uncertain_Samples": uncertain_count,
            "Coverage_Percent": round(coverage_pct, 1),
            "Verified_HC_Precision": round(hc_precision, 3) if not np.isnan(hc_precision) else "N/A",
            "Confident_Accuracy": round(accuracy, 3) if not np.isnan(accuracy) else "N/A"
        })
