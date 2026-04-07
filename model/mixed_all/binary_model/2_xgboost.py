# ========================================================
# IMPORT STATIC CONFIGURATIONS AND DATA LOADING FUNCTION
# ========================================================
import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import (roc_auc_score, accuracy_score, recall_score, roc_curve, 
                             average_precision_score, f1_score, matthews_corrcoef)
from sklearn.model_selection import cross_val_predict, RandomizedSearchCV, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

# Import custom configurations
from model.pipeline_config import (preprocessing_pipelines, feature_pipelines, 
                                   scoring_metrics, load_nirs_data)

# Import the data
X_train, X_test, y_train_binary, y_test_binary, train_country, test_country = load_nirs_data(
    splits=10, threshold=1000, weighted=False
)

# Define the sources and targets for iteration
sources = ['intact', 'ground']
targets = ['DON', 'Total_DON']

# ==========================================
# XGBOOST MODEL OPTIMIZATION
# ==========================================
# Parameter distribution for RandomizedSearchCV
xgb_param_dist = {
    'xgb__n_estimators': [50, 100, 200, 400, 600, 1000],
    'xgb__learning_rate': [0.005, 0.01, 0.03, 0.05, 0.1, 0.2],
    'xgb__max_depth': [2, 3, 4, 5, 6, 8],
    'xgb__subsample': [0.6, 0.8, 1.0],
    'xgb__colsample_bytree': [0.6, 0.8, 1.0],
    'xgb__colsample_bylevel': [0.6, 0.8, 1.0],
    'xgb__gamma': [0, 0.01, 0.1, 0.5, 1, 2],
    'xgb__min_child_weight': [1, 3, 5, 7, 10],
    'xgb__reg_alpha': [0, 0.01, 0.1, 1, 10],
    'xgb__reg_lambda': [0.1, 1, 5, 10, 20]
}

xgb_experiment_results = [] 

# Main nested CV loop
for source in sources:
    for target in targets:
        for i in range(10): 
            print(f"\nEvaluating: {source} samples for {target} (Split {i+1}/10)")
            
            # Extract raw data arrays
            X_tr_spectra = X_train[source][i].values
            X_te_spectra = X_test[source][i].values
            
            # Map 'HC' to 1 and 'LC' to 0
            y_tr = np.where(y_train_binary[source][i][target].values == 'HC', 1, 0)
            y_te = np.where(y_test_binary[source][i][target].values == 'HC', 1, 0)
            
            for pipe_name, preprocessor in preprocessing_pipelines.items():
                for fp_name, fp_method in feature_pipelines.items():
                    print(f"  -> Running Prep: {pipe_name} | Feature: {fp_name}...", end=" ")
                    start_time = time.time()
                    
                    try:
                        # --------------------------------------------------------
                        # BUILD THE LEAK-PROOF PIPELINE
                        # --------------------------------------------------------
                        steps = []
                        
                        if preprocessor is not None:
                            steps.append(('prep', preprocessor))
                            
                        if fp_method is not None:
                            steps.append(('feature_selection', fp_method))
                            
                        # Calculate scale_pos_weight to handle class imbalance dynamically
                        num_neg = np.sum(y_tr == 0)
                        num_pos = np.sum(y_tr == 1)
                        scale_weight = num_neg / num_pos if num_pos > 0 else 1.0

                        # Base XGB model
                        base_xgb = XGBClassifier(
                            scale_pos_weight=scale_weight, 
                            eval_metric='logloss',
                            random_state=270725
                        )
                        steps.append(('xgb', base_xgb))
                        
                        pipeline = Pipeline(steps)
                        
                        # --------------------------------------------------------
                        # INNER LOOP: HYPERPARAMETER TUNING
                        # --------------------------------------------------------
                        random_search = RandomizedSearchCV(
                            estimator=pipeline,
                            param_distributions=xgb_param_dist,
                            n_iter=20,          
                            cv=10,
                            scoring=scoring_metrics,
                            refit='roc_auc',
                            n_jobs=-1,          
                            random_state=270725
                        )
                        
                        # Pass RAW spectra directly to search
                        random_search.fit(X_tr_spectra, y_tr)
                        
                        # --------------------------------------------------------
                        # OUT-OF-FOLD CALIBRATION & THRESHOLD OPTIMIZATION
                        # --------------------------------------------------------
                        best_pipeline = random_search.best_estimator_
                        calibrated_xgb = CalibratedClassifierCV(best_pipeline, cv=10, method='sigmoid')

                        cv_strategy = StratifiedKFold(n_splits=10, shuffle=True, random_state=270725)
                        calibrated_cv_probs = cross_val_predict(
                            calibrated_xgb, X_tr_spectra, y_tr, 
                            cv=cv_strategy, method='predict_proba', n_jobs=-1
                        )
                        
                        # Youden's J Threshold
                        fpr, tpr, thresholds = roc_curve(y_tr, calibrated_cv_probs[:, 1])
                        best_idx = np.argmax(tpr - fpr)
                        optimal_threshold = thresholds[best_idx]
                        
                        calibrated_cv_preds = (calibrated_cv_probs[:, 1] >= optimal_threshold).astype(int)
                        
                        true_cv_auc = roc_auc_score(y_tr, calibrated_cv_probs[:, 1])
                        true_cv_accuracy = accuracy_score(y_tr, calibrated_cv_preds)
                        true_cv_sensitivity = recall_score(y_tr, calibrated_cv_preds)
                        true_cv_specificity = recall_score(y_tr, calibrated_cv_preds, pos_label=0)
                        
                        # --------------------------------------------------------
                        # OUTER LOOP: FINAL EXAMINAMTION ON TEST SET
                        # --------------------------------------------------------
                        calibrated_xgb.fit(X_tr_spectra, y_tr)
                        
                        test_probs = calibrated_xgb.predict_proba(X_te_spectra)[:, 1]
                        test_preds = (test_probs >= optimal_threshold).astype(int)

                        test_auc = roc_auc_score(y_te, test_probs)
                        test_accuracy = accuracy_score(y_te, test_preds)
                        test_sensitivity = recall_score(y_te, test_preds)
                        test_specificity = recall_score(y_te, test_preds, pos_label=0, zero_division=0)
                        test_PR_curve = average_precision_score(y_te, test_preds)
                        test_f1 = f1_score(y_te, test_preds, zero_division=0)
                        test_mcc = matthews_corrcoef(y_te, test_preds)
                        
                        elapsed_time = time.time() - start_time
                        print(f"Done! (Test AUC: {test_auc:.3f} | True CV Acc: {true_cv_accuracy:.3f})")
                        
                        # --------------------------------------------------------
                        # 5. SAVE RESULTS
                        # --------------------------------------------------------
                        xgb_experiment_results.append({
                            'Sample_Source': source,
                            'Target_Toxin': target,
                            'Split': i,
                            'Preprocessing': pipe_name,
                            'Feature_Method': fp_name,
                            'Best_n_estimators': random_search.best_params_['xgb__n_estimators'],
                            'Best_learning_rate': random_search.best_params_['xgb__learning_rate'],
                            'Best_max_depth': random_search.best_params_['xgb__max_depth'],
                            'Best_subsample': random_search.best_params_['xgb__subsample'],
                            'Best_colsample_bytree': random_search.best_params_['xgb__colsample_bytree'],
                            'Best_gamma': random_search.best_params_['xgb__gamma'],
                            'Best_Colsample_bylevel': random_search.best_params_['xgb__colsample_bylevel'],
                            'Best_min_child_weight': random_search.best_params_['xgb__min_child_weight'],
                            'Best_reg_alpha': random_search.best_params_['xgb__reg_alpha'],
                            'Best_reg_lambda': random_search.best_params_['xgb__reg_lambda'],
                            'Optimal_Threshold': optimal_threshold,
                            'True_CV_AUC': true_cv_auc,
                            'True_CV_Accuracy': true_cv_accuracy,
                            'True_CV_Sensitivity': true_cv_sensitivity,
                            'True_CV_Specificity': true_cv_specificity,
                            'Test_AUC': test_auc,
                            'Test_Accuracy': test_accuracy,
                            'Test_Sensitivity': test_sensitivity,
                            'Test_Specificity': test_specificity,
                            'Test_PR_Curve': test_PR_curve,
                            'Test_F1': test_f1,
                            'Test_MCC': test_mcc,
                            'Time_Sec': np.round(elapsed_time, 2)
                        })
                        
                    except Exception as e:
                        print(f"Failed. Error: {str(e)}")

# Combine all results and export
optimizied_algorithm = pd.DataFrame(xgb_experiment_results)
optimizied_algorithm.to_csv('./results/mixed_all/2_XGB_results.csv', index=False)
print("\nAll combinations processed successfully! XGBoost data saved to CSV.")