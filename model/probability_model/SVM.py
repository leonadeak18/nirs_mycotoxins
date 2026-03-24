import pandas as pd
import numpy as np
import time
from sklearn.model_selection import cross_val_predict, StratifiedKFold, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_recall_curve, roc_auc_score

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
        ('smoothing', SavitzkyGolay(window_size=15, polynomial_order=2, derivate_order = 0))
    ]),
    "SNV_Smoothing": Pipeline([
        ('snv', ch.StandardNormalVariate()),
        ('smoothing', SavitzkyGolay(window_size=15, polynomial_order=2, derivate_order = 0))
    ]),
    "EMSC_Smoothing": Pipeline([
        ('emsc', ch.ExtendedMultiplicativeScatterCorrection()),
        ('smoothing', SavitzkyGolay(window_size=15, polynomial_order=2, derivate_order = 0))
    ]),
    "1st_derivative": Pipeline([
        ('first_derivative', SavitzkyGolay(window_size=15, polynomial_order=2, derivate_order = 1))
    ]),
    "2nd_derivative": Pipeline([
        ('second_derivative', SavitzkyGolay(window_size=15, polynomial_order=2, derivate_order = 2))
    ]),
    "RNV_Smoothing": Pipeline([
        ('rnv', ch.RobustNormalVariate()),
        ('smoothing', SavitzkyGolay(window_size=15, polynomial_order=2, derivate_order = 0))
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
                    random_search.fit(X_tr_final, y_tr)
                    
                    # 2. Extract the winning SVM and Calibrate it
                    best_svm = random_search.best_estimator_
                    best_svm.probability = True # Required for calibration
                    
                    calibrated_svm = CalibratedClassifierCV(best_svm, cv=3)
                    calibrated_svm.fit(X_tr_final, y_tr)
                    
                    # 3. Test on the unseen Test Set
                    test_probs = calibrated_svm.predict_proba(X_te_final)[:, 1]
                    test_auc = roc_auc_score(y_te, test_probs)
                    
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

# ==========================================
# ROBUST THRESHOLD FINDING FOR SVM
# ==========================================
all_results = []
calibration_methods = ["sigmoid", "isotonic"]
n_repeats = 1000
target_precision = 0.95

optimized_model = {
    "intact_DON": Pipeline ([
        ('preprocessing', preprocessing_pipelines["RNV_Smoothing"]),
        ('feature_selection', feature_pipelines["Factor_Analysis"]),
        ('svc', SVC(kernel='rbf', C=10, gamma=0.001, class_weight='balanced', random_state=270725, probability=True))
        ]),
    "intact_Total_DON": Pipeline([
        ('preprocessing', preprocessing_pipelines["RNV_Smoothing"]),
        ('feature_selection', feature_pipelines["Factor_Analysis"]),
        ('svc', SVC(kernel='rbf', C=5, gamma=0.01, class_weight='balanced', random_state=270725, probability=True))
        ]),
    "ground_DON": Pipeline([
        ('preprocessing', preprocessing_pipelines["RNV_Smoothing"]),
        ('feature_selection', feature_pipelines["RFE"]),
        ('svc', SVC(kernel='rbf', C=1000, gamma='auto', class_weight='balanced', random_state=270725, probability=True))
        ]),
    "ground_Total_DON": Pipeline([
        ('preprocessing', preprocessing_pipelines["RNV_Smoothing"]),
        ('feature_selection', feature_pipelines["ANOVA_Pencentile"]),
        ('svc', SVC(kernel='rbf', C=1000, gamma=0.001, class_weight='balanced', random_state=270725, probability=True))
        ])
}


base_pipeline = optimized_model["ground_DON"]
pipe_name = list(base_pipeline.named_steps.keys())[0]
fp_name = list(base_pipeline.named_steps.keys())[1]

for source in sources:
    for target in targets:
        print(f"Evaluating: {source} with {target}")
        
        # Extract the specific data arrays for this loop iteration
        X_tr = X_train[source].values
        X_te = X_test[source].values
        
        # Ensure target variables are 1D arrays
        y_tr = y_train[source][target].values
        y_te = y_test[source][target].values
        dataset_key = f"{source}_{target}"
        
        if dataset_key not in optimized_model:
            continue
        
        base_pipeline = optimized_model[dataset_key]
        pipe_name = list(base_pipeline.named_steps.keys())[0]
        fp_name = list(base_pipeline.named_steps.keys())[1]

        for method in calibration_methods:
            print(f"  -> Applying Calibration with {method} method...")
            calibrated_model = CalibratedClassifierCV(base_pipeline, cv=10, method=method)

            upper_thresholds = []
            lower_thresholds = []

            for i in range(n_repeats):
                cv_strategy = StratifiedKFold(n_splits=10, shuffle=True, random_state=i)
                cv_probs = cross_val_predict(calibrated_model, X_tr, y_tr, cv=cv_strategy, method='predict_proba', n_jobs=-1)
                
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
            })
