import pandas as pd
import numpy as np
import time
from sklearn.model_selection import cross_val_predict, StratifiedKFold, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_recall_curve, roc_auc_score, accuracy_score, precision_score
from sklearn.base import clone

# ==========================================
# IMPORT THE DATA
# ==========================================
from nirs.nirs_dataset import X_train, y_train, X_test, y_test
from utils.preparation import binary_transform_with_weight

# Define the scopes
sources = ['intact', 'ground']
targets = ['DON', 'Total_DON']

## remove Log_Total_DON and Log_Total_DON columns from y_test
for key in y_test:
    y_test[key] = y_test[key].drop(columns=['Log_DON', 'Log_Total_DON'])
    y_train[key] = y_train[key].drop(columns=['Log_DON', 'Log_Total_DON'])

## Categorize into 2 classes based on quantiles
for source in sources:
    y_test[source], y_train[source] = binary_transform_with_weight(y_test[source]), binary_transform_with_weight(y_train[source])


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
# ROBUST THRESHOLD FINDING FOR SVM
# ==========================================
optimized_model = {
    # "intact_DON": Pipeline ([
    #     ('1st_derivative', preprocessing_pipelines["1st_derivative"]),
    #     ('ANOVA_FDR', feature_pipelines["ANOVA_FDR"]),
    #     ('svc', SVC(kernel='rbf', C=500, gamma='scale', class_weight='balanced', random_state=270725, probability=True))
    #     ]),
    "intact_Total_DON": Pipeline([
        ('2nd_derivative', preprocessing_pipelines["2nd_derivative"]),
        ('ANOVA_FDR', feature_pipelines["ANOVA_FDR"]),
        ('svc', SVC(kernel='rbf', C=100, gamma='scale', class_weight='balanced', random_state=270725, probability=True))
        ]),
    # "ground_DON": Pipeline([
    #     ('1st_derivative', preprocessing_pipelines["1st_derivative"]),
    #     ('Factor_Analysis', feature_pipelines["Factor_Analysis"]),
    #     ('svc', SVC(kernel='rbf', C=0.1, gamma=0.01, class_weight='balanced', random_state=270725, probability=True))
    #     ]),
    "ground_Total_DON": Pipeline([
        ('SNV_Smoothing', preprocessing_pipelines["SNV_Smoothing"]),
        ('ANOVA_FDR', feature_pipelines["ANOVA_FDR"]),
        ('svc', SVC(kernel='rbf', C=1000, gamma=0.01, class_weight='balanced', random_state=270725, probability=True))
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
        dataset_key = f"{source}_{target}"
        if dataset_key not in optimized_model:
            continue

        production_models[dataset_key] = {}
        print(f"Processing: {dataset_key}")
        
        # Extract the specific data arrays for this loop iteration
        X_tr = X_train[source].values
        
        # Ensure target variables are 1D arrays
        y_tr = y_train[source][target].values

        # Convert HC to 1 and LC to 0 for precision-recall calculations
        y_tr = np.where(y_tr == 'HC', 1, 0)

        #samples weight
        tr_weights = y_train[source][target + '_weight'].values

        base_pipeline = optimized_model[dataset_key]
        pipe_name = list(base_pipeline.named_steps.keys())[0]
        fp_name = list(base_pipeline.named_steps.keys())[1]

        base_svm = base_pipeline.named_steps['svc']

        for method in calibration_methods:
            threshold_results[source][target][method] = {}
            print(f"  -> Applying Calibration with {method} method...")
            calibrated_model = CalibratedClassifierCV(base_svm, cv=10, method=method)
            
            calibrated_pipeline = clone(base_pipeline)
            calibrated_pipeline.steps[-1] = ('calibrated_svc', calibrated_model)
            
            upper_thresholds = []
            lower_thresholds = []

            for i in range(n_repeats):
                cv_strategy = StratifiedKFold(n_splits=10, shuffle=True, random_state=i)
                cv_probs = cross_val_predict(
                    calibrated_pipeline, X_tr, y_tr, 
                    cv=cv_strategy, method='predict_proba', n_jobs=-1,
                    params={'calibrated_svc__sample_weight': tr_weights})
                
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
            final_fitted_model = calibrated_pipeline.fit(X_tr, y_tr, calibrated_svc__sample_weight=tr_weights)
            production_models[dataset_key][method] = {
                "model": final_fitted_model,
                "lower_threshold": np.median(lower_thresholds),
                "upper_threshold": np.median(upper_thresholds)
            }

## Export all important results
### export the model
import joblib
joblib.dump(production_models, "./model/probability_model/svm_models.pkl")


import os
# Create a folder for the outputs if it doesn't exist yet
output_folder = "./model/probability_model/"
os.makedirs(output_folder, exist_ok=True)

#Export all results
summary_df = pd.DataFrame(all_results)
summary_path = os.path.join(output_folder, "threshold_summary_results.csv")
summary_df.to_csv(summary_path, index=False)
print(f"✨ Successfully saved summary to: {summary_path}")

## Export threshold_results
flattened_raw_data = []

# Loop through the nested dictionary to unpack the lists
for source, targets_dict in threshold_results.items():
    for target, methods_dict in targets_dict.items():
        for method, threshold_types in methods_dict.items():
            
            # Extract the raw lists
            upper_list = threshold_types.get('upper', [])
            lower_list = threshold_types.get('lower', [])
            
            # Unpack them row by row
            for i in range(len(upper_list)):
                flattened_raw_data.append({
                    "Material": source,
                    "Target": target,
                    "Calibration_Method": method,
                    "CV_Repeat_Number": i + 1,
                    "Lower_Threshold": lower_list[i] if i < len(lower_list) else None,
                    "Upper_Threshold": upper_list[i] if i < len(upper_list) else None
                })

raw_df = pd.DataFrame(flattened_raw_data)
raw_path = os.path.join(output_folder, "raw_threshold_arrays.csv")
raw_df.to_csv(raw_path, index=False)








# ==========================================
# PLOTTING THE DISTRIBUTIONS OF THRESHOLDS
# ==========================================
from utils.plotting import plot_threshold_distribution
df = pd.read_csv("./model/probability_model/raw_threshold_arrays.csv")
plot_df = df[(df['Material'] == 'intact') & (df['Target'] == 'Total_DON') & (df['Calibration_Method'] == 'sigmoid')]
plot = plot_threshold_distribution(plot_df, title="Distribution of CV Probability Thresholds for Intact Total DON (Sigmoid Calibration)")
plot.show()





# ==========================================
# VERIFICATION ON TEST SET
# ==========================================
import joblib
import numpy as np

# Load the models
production_models = joblib.load("./model/probability_model/production_models.pkl")
registry_entry = production_models['ground_Total_DON']["isotonic"]

# This 'model' is your FULL fitted pipeline
model = registry_entry["model"] 
lower_t = registry_entry["lower_threshold"]
upper_t = registry_entry["upper_threshold"]

# 1. Get the RAW test data
X_te_raw = X_test['ground'].values

# Set up your target variables
y_te_binary = np.where(y_test['ground']['Total_DON'].values == 'HC', 1, 0)
y_te_strings = np.where(y_te_binary == 1, 'HC', 'LC')

# 2. Pass the RAW data directly into the pipeline
# The pipeline will automatically apply RNV_Smoothing and Factor_Analysis for you
probs = model.predict_proba(X_te_raw)[:, 1]

predictions = np.full(probs.shape, 'UC', dtype=object)
predictions[probs < lower_t] = 'LC'
predictions[probs > upper_t] = 'HC'

# 4. Filter for Confident Predictions
eval_df = pd.DataFrame({'True_Label': y_test['ground']['Total_DON'].values, 'Prediction': predictions}, index=X_test['ground'].index)
confident_df = eval_df[eval_df['Prediction'] != 'UC']

uncertain_count = len(eval_df) - len(confident_df)
coverage_pct = (len(confident_df) / len(eval_df)) * 100

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
