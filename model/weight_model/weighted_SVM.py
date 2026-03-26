from model.probability_model.SVM import cv_strategy
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import cross_val_predict, StratifiedKFold, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import (roc_auc_score, accuracy_score, recall_score, confusion_matrix, make_scorer, roc_curve)


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
        # C=1 applies moderate regularization to drop some features
        ('Lasso_L1', SelectFromModel(LogisticRegression(penalty='l1', solver='liblinear', C=1, random_state=270725)))
    ]),
    
    "RFE": Pipeline([
        # step=0.1 means it drops 5% of features at a time to run faster
        ('RFE', RFE(estimator=SVC(kernel='linear', random_state=270726), n_features_to_select= 0.2, step=0.05))
    ])
}

# ==========================================
# SVM MODEL OPTIMIZATION
# ==========================================
#define specifity make_scorer for RandomizedSearchCV
def specificity_score(y_true, y_pred, sample_weight=None):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, sample_weight=sample_weight).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0

# create a scoring dictionary
scoring_metrics = {
    'roc_auc': make_scorer(roc_auc_score, response_method="predict_proba"),
    'accuracy': make_scorer(accuracy_score),
    'sensitivity': make_scorer(recall_score), # Sensitivity is the same as Recall
    'specificity': make_scorer(specificity_score)
}
wsvm_experiment_results = []
# Define the grid of parameters for the SVM to test
wsvm_experiment_results = []
# Define the grid of parameters for the SVM to test
svm_param_dist = {
    'C': [0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
}

for source in sources:
    for target in targets:
        print(f"Evaluating: {source} samples for {target}")
        
        # Extract the specific data arrays
        X_tr = X_train[source].values
        X_te = X_test[source].values
        
        # Ensure target variables are 1D arrays and map 'HC' to 1 and 'LC' to 0
        y_tr = np.where(y_train[source][target].values == 'HC', 1, 0)
        y_te = np.where(y_test[source][target].values == 'HC', 1, 0)

        # Extract sample weights
        tr_weights = y_train[source][target + '_weight'].values
        te_weights = y_test[source][target + '_weight'].values
        
        for pipe_name, preprocessor in preprocessing_pipelines.items():
            # Apply Scatter Corrections / Derivatives
            X_train_prep = preprocessor.fit_transform(X_tr)
            X_test_prep = preprocessor.transform(X_te)

            for fp_name, fp_method in feature_pipelines.items():
                print(f"Running -> Prep: {pipe_name} | Feature: {fp_name}...", end=" ")
                start_time = time.time()
                
                try:
                    # Apply Feature Extraction / Selection
                    if fp_method is not None:
                        X_tr_final = fp_method.fit_transform(X_train_prep, y_tr)
                        X_te_final = fp_method.transform(X_test_prep)
                    else:
                        X_tr_final = X_train_prep
                        X_te_final = X_test_prep
                        
                    # Skip if a selector dropped ALL features
                    if X_tr_final.shape[1] == 0:
                        print("Skipped (0 features selected).")
                        continue

                    # Define base SVC model
                    base_svm = SVC(kernel='rbf', class_weight='balanced', random_state=270725, probability=True)
                    
                    # Run Randomized Search to find the best C and Gamma
                    random_search = RandomizedSearchCV(
                        estimator=base_svm,
                        param_distributions=svm_param_dist,
                        n_iter=50,          
                        cv=10,
                        scoring=scoring_metrics,
                        refit='roc_auc',
                        n_jobs=-1,          
                        random_state=270725
                    )
                    random_search.fit(X_tr_final, y_tr, sample_weight=tr_weights)
                    
                    # Extract the winning SVM from the search and wrap in calibrator
                    best_svm = random_search.best_estimator_
                    best_svm.probability = True 
                    calibrated_svm = CalibratedClassifierCV(best_svm, cv=10, method='sigmoid')
                    
                    # Get true, out-of-fold calibrated probabilities for the training set
                    cv_strategy = StratifiedKFold(n_splits=10, shuffle=True, random_state=270725)
                    calibrated_cv_probs = cross_val_predict(
                        calibrated_svm, X_tr_final, y_tr, 
                        cv=cv_strategy, method='predict_proba', n_jobs=-1,
                        params={'sample_weight': tr_weights} 
                    )
                    
                    # Find the optimal threshold using Youden's J
                    fpr, tpr, thresholds = roc_curve(y_tr, calibrated_cv_probs[:, 1], sample_weight=tr_weights)
                    best_idx = np.argmax(tpr - fpr)
                    optimal_threshold = thresholds[best_idx]
                    
                    # Convert probabilities to hard predictions using your optimized threshold
                    calibrated_cv_preds = (calibrated_cv_probs[:, 1] >= optimal_threshold).astype(int)
                    
                    # Calculate true CV metrics
                    true_cv_auc = roc_auc_score(y_tr, calibrated_cv_probs[:, 1], sample_weight=tr_weights)
                    true_cv_accuracy = accuracy_score(y_tr, calibrated_cv_preds, sample_weight=tr_weights)
                    true_cv_sensitivity = recall_score(y_tr, calibrated_cv_preds, sample_weight=tr_weights)
                    true_cv_specificity = recall_score(y_tr, calibrated_cv_preds, pos_label=0, sample_weight=tr_weights)
                    
                    # 6. Fit the calibrator on the FULL training set for final testing
                    calibrated_svm.fit(X_tr_final, y_tr, sample_weight=tr_weights)
                    
                    # 7. Test on the Test Set
                    test_probs = calibrated_svm.predict_proba(X_te_final)[:, 1]
                    
                    # 8. Apply your optimal threshold to the test set predictions
                    test_preds = (test_probs >= optimal_threshold).astype(int)

                    # 9. Extract accurate test metrics
                    test_auc = roc_auc_score(y_te, test_probs, sample_weight=te_weights)
                    test_accuracy = accuracy_score(y_te, test_preds, sample_weight=te_weights)
                    test_sensitivity = recall_score(y_te, test_preds, sample_weight=te_weights)
                    test_specificity = recall_score(y_te, test_preds, pos_label=0, sample_weight=te_weights)
                    
                    elapsed_time = time.time() - start_time
                    print(f"Done! (Test AUC: {test_auc:.3f} | True CV Acc: {true_cv_accuracy:.3f})")
                    
                    # 10. Save the aligned results!
                    wsvm_experiment_results.append({
                        'Sample_Source': source,
                        'Target_Toxin': target,
                        'Preprocessing': pipe_name,
                        'Feature_Method': fp_name,
                        'Best_C': random_search.best_params_['C'],
                        'Best_Gamma': random_search.best_params_['gamma'],
                        'Optimal_Threshold': optimal_threshold,
                        'True_CV_AUC': true_cv_auc,
                        'True_CV_Accuracy': true_cv_accuracy,
                        'True_CV_Sensitivity': true_cv_sensitivity,
                        'True_CV_Specificity': true_cv_specificity,
                        'Test_AUC': test_auc,
                        'Test_Accuracy': test_accuracy,
                        'Test_Sensitivity': test_sensitivity,
                        'Test_Specificity': test_specificity,
                        'Time_Sec': np.round(elapsed_time, 2)
                    })
                    
                except Exception as e:
                    print(f"Failed. Error: {str(e)}")
#combine all results into a dataframe for easier analysis
optimizied_algorithm = pd.DataFrame(wsvm_experiment_results)
optimizied_algorithm.to_csv('./results/weighted_SVM_results.csv', index=False)
