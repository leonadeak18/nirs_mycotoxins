from utils.preparation import binary_transform_with_weight
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import cross_val_predict, StratifiedKFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import (roc_auc_score, accuracy_score, recall_score, confusion_matrix, make_scorer)

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
        ('RFE', RFE(estimator=SVC(kernel='linear', random_state=270726), n_features_to_select= 0.2, step=0.05))
    ])
}

# ==========================================
# RF MODEL OPTIMIZATION
# ==========================================
#define specifity make_scorer for RandomizedSearchCV
def specificity_score(y_true, y_pred, sample_weight=None):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, sample_weight=sample_weight).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0

# create a scoring dictionary
scoring_metrics = {
    'roc_auc': make_scorer(roc_auc_score, needs_proba=True),
    'accuracy': make_scorer(accuracy_score),
    'sensitivity': make_scorer(recall_score), # Sensitivity is the same as Recall
    'specificity': make_scorer(specificity_score)
}

# Define the grid of parameters for the Random Forest to test
rf_param_dist = {
    'n_estimators': [50, 100, 200, 300, 400, 500, 700, 1000],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10, 20, 50],
    'min_samples_leaf': [1, 2, 4, 10, 20]
}

wrf_experiment_results = []
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

                    # Define base RF model
                    base_rf = RandomForestClassifier(class_weight='balanced', random_state=270725)
                    
                    # Run Randomized Search to find the best hyperparameters
                    random_search = RandomizedSearchCV(
                        estimator=base_rf,
                        param_distributions=rf_param_dist,
                        n_iter=50,          
                        cv=10,
                        scoring=scoring_metrics,
                        refit='roc_auc',  
                        n_jobs=-1,          
                        random_state=270725
                    )
                    random_search.fit(X_tr_final, y_tr, sample_weight=tr_weights)
                    
                    # Extract the winning RF and Calibrate it
                    best_rf = random_search.best_estimator_
                    best_rf.probability = True
                    
                    calibrated_rf = CalibratedClassifierCV(best_rf, cv=10)
                    calibrated_rf.fit(X_tr_final, y_tr, sample_weight=tr_weights)

                    # Extract CV metrics for the best model
                    best_idx = random_search.best_index_
                    cv_auc = random_search.cv_results_['mean_test_roc_auc'][best_idx]
                    cv_accuracy = random_search.cv_results_['mean_test_accuracy'][best_idx]
                    cv_sensitivity = random_search.cv_results_['mean_test_sensitivity'][best_idx]
                    cv_specificity = random_search.cv_results_['mean_test_specificity'][best_idx]

                    
                    # Test on the Test Set
                    test_probs = calibrated_rf.predict_proba(X_te_final)[:, 1]
                    test_preds = calibrated_rf.predict(X_te_final)

                    # extract test metrics
                    test_auc = roc_auc_score(y_te, test_probs, sample_weight=te_weights)
                    test_accuracy = accuracy_score(y_te, test_preds, sample_weight=te_weights)
                    test_sensitivity = recall_score(y_te, test_preds, sample_weight=te_weights)
                    test_specificity = specificity_score(y_te, test_preds, sample_weight=te_weights)
                    
                    elapsed_time = time.time() - start_time
                    print(f"Done! (Test AUC: {test_auc:.3f})")
                    
                    # 4. Save the results
                    wrf_experiment_results.append({
                        'Sample_Source': source,
                        'Target_Toxin': target,
                        'Preprocessing': pipe_name,
                        'Feature_Method': fp_name,
                        'Best_n_estimators': random_search.best_params_['n_estimators'],
                        'Best_max_depth': random_search.best_params_['max_depth'],
                        'Best_max_features': random_search.best_params_['max_features'],
                        'Best_min_samples_split': random_search.best_params_['min_samples_split'],
                        'Best_min_samples_leaf': random_search.best_params_['min_samples_leaf'],
                        'CV_Val_AUC': cv_auc,
                        'CV_Val_Accuracy': cv_accuracy,
                        'CV_Val_Sensitivity': cv_sensitivity,
                        'CV_Val_Specificity': cv_specificity,
                        'Test_AUC': test_auc,
                        'Test_Accuracy': test_accuracy,
                        'Test_Sensitivity': test_sensitivity,
                        'Test_Specificity': test_specificity,
                        'Time_Sec': np.round(elapsed_time, 2)
                    })
                    
                except Exception as e:
                    print(f"Failed. Error: {str(e)}")
#combine all results into a dataframe for easier analysis
optimizied_RF = pd.DataFrame(wrf_experiment_results)
optimizied_RF.to_csv('./results/weighted_RF_results.csv', index=False)
