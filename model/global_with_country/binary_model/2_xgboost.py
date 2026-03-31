# ========================================================
# IMPORT STATIC CONFIGURATIONS AND DATA LOADING FUNCTION
# ========================================================
import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import (roc_auc_score, accuracy_score, recall_score, roc_curve)
from sklearn.model_selection import cross_val_predict, RandomizedSearchCV, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from model.pipeline_config import (preprocessing_pipelines, feature_pipelines, scoring_metrics, load_nirs_data)
from xgboost import XGBClassifier

#import the data
X_train, X_test, y_train_binary, y_test_binary, train_country, test_country = load_nirs_data(splits=10, threshold=1000, weighted=False)

# define the sources and targets for iteration
sources = ['intact', 'ground']
targets = ['DON', 'Total_DON']


# ==========================================
# XGBOOST MODEL OPTIMIZATION
# ==========================================
# Define the grid of parameters for XGBoost to test
xgb_param_dist = {
    'n_estimators': [50, 100, 200, 400, 600, 1000],
    'learning_rate': [0.005, 0.01, 0.03, 0.05, 0.1, 0.2],
    'max_depth': [2, 3, 4, 5, 6, 8],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'colsample_bylevel': [0.6, 0.8, 1.0],
    'gamma': [0, 0.01, 0.1, 0.5, 1, 2],
    'min_child_weight': [1, 3, 5, 7, 10],
    'reg_alpha': [0, 0.01, 0.1, 1, 10],
    'reg_lambda': [0.1, 1, 5, 10, 20]
}

# Initialize the encoder for the countries
country_encoder = OrdinalEncoder()
xgb_experiment_results = [] 

for source in sources:
    for target in targets:
        for i in range(10): 
            print(f"Evaluating: {source} samples for {target} (Split {i})")
            
            # Extract the specific data arrays for this specific split
            X_tr_spectra = X_train[source][i].values
            X_te_spectra = X_test[source][i].values
            
            # Explicitly map 'HC' to 1 and 'LC' to 0
            y_tr = np.where(y_train_binary[source][i][target].values == 'HC', 1, 0)
            y_te = np.where(y_test_binary[source][i][target].values == 'HC', 1, 0)
            
            # Extract country text and reshape for the encoder
            country_tr_raw = train_country[source][i].to_numpy().reshape(-1, 1)
            country_te_raw = test_country[source][i].to_numpy().reshape(-1, 1)
            
            # Convert text (Austria/France) to numbers (0/1)
            country_tr_encoded = country_encoder.fit_transform(country_tr_raw)
            country_te_encoded = country_encoder.transform(country_te_raw)
            
            for pipe_name, preprocessor in preprocessing_pipelines.items():
                # Apply Scatter Corrections / Derivatives ONLY to the spectra
                X_train_prep = preprocessor.fit_transform(X_tr_spectra)
                X_test_prep = preprocessor.transform(X_te_spectra)

                for fp_name, fp_method in feature_pipelines.items():
                    print(f"Running -> Prep: {pipe_name} | Feature: {fp_name}...", end=" ")
                    start_time = time.time()
                    
                    try:
                        # Apply Feature Extraction / Selection ONLY to the spectra
                        if fp_method is not None:
                            X_tr_final_spectra = fp_method.fit_transform(X_train_prep, y_tr)
                            X_te_final_spectra = fp_method.transform(X_test_prep)
                        else:
                            X_tr_final_spectra = X_train_prep
                            X_te_final_spectra = X_test_prep
                            
                        # Skip if a selector dropped ALL features
                        if X_tr_final_spectra.shape[1] == 0:
                            print("Skipped (0 features selected).")
                            continue

                        # Glue the 0/1 country column onto the right side of the processed spectra
                        X_tr_final = np.hstack((X_tr_final_spectra, country_tr_encoded))
                        X_te_final = np.hstack((X_te_final_spectra, country_te_encoded))

                        # Calculate scale_pos_weight to handle class imbalance dynamically
                        num_neg = np.sum(y_tr == 0)
                        num_pos = np.sum(y_tr == 1)
                        scale_weight = num_neg / num_pos if num_pos > 0 else 1.0

                        # Define base XGB model
                        base_xgb = XGBClassifier(
                            scale_pos_weight=scale_weight, 
                            eval_metric='logloss',
                            random_state=270725
                        )
                        
                        # Run Randomized Search to find the best parameters on the COMBINED data
                        random_search = RandomizedSearchCV(
                            estimator=base_xgb,
                            param_distributions=xgb_param_dist,
                            n_iter=20,          
                            cv=10,
                            scoring=scoring_metrics,
                            refit='roc_auc',
                            n_jobs=-1,          
                            random_state=270725
                        )
                        random_search.fit(X_tr_final, y_tr)
                        
                        # Extract the winning XGB model and Calibrate it
                        best_xgb = random_search.best_estimator_
                        best_xgb.probability = True
                        calibrated_xgb = CalibratedClassifierCV(best_xgb, cv=10, method='sigmoid')

                        # Get true, out-of-fold calibrated probabilities for the training set
                        cv_strategy = StratifiedKFold(n_splits=10, shuffle=True, random_state=270725)
                        calibrated_cv_probs = cross_val_predict(
                            calibrated_xgb, X_tr_final, y_tr, 
                            cv=cv_strategy, method='predict_proba', n_jobs=-1
                        )
                        
                        # Find the optimal threshold using Youden's J
                        fpr, tpr, thresholds = roc_curve(y_tr, calibrated_cv_probs[:, 1])
                        best_idx = np.argmax(tpr - fpr)
                        optimal_threshold = thresholds[best_idx]
                        
                        # Convert probabilities to hard predictions using your optimized threshold
                        calibrated_cv_preds = (calibrated_cv_probs[:, 1] >= optimal_threshold).astype(int)
                        
                        # Calculate true CV metrics
                        true_cv_auc = roc_auc_score(y_tr, calibrated_cv_probs[:, 1])
                        true_cv_accuracy = accuracy_score(y_tr, calibrated_cv_preds)
                        true_cv_sensitivity = recall_score(y_tr, calibrated_cv_preds)
                        true_cv_specificity = recall_score(y_tr, calibrated_cv_preds, pos_label=0)
                        
                        # Fit the calibrator on the FULL training set for final testing
                        calibrated_xgb.fit(X_tr_final, y_tr)
                        
                        # Test on the Test Set
                        test_probs = calibrated_xgb.predict_proba(X_te_final)[:, 1]
                        
                        # Apply your optimal threshold to the test set predictions
                        test_preds = (test_probs >= optimal_threshold).astype(int)

                        # Extract accurate test metrics
                        test_auc = roc_auc_score(y_te, test_probs)
                        test_accuracy = accuracy_score(y_te, test_preds)
                        test_sensitivity = recall_score(y_te, test_preds)
                        test_specificity = recall_score(y_te, test_preds, pos_label=0)
                        
                        elapsed_time = time.time() - start_time
                        print(f"Done! (Test AUC: {test_auc:.3f} | True CV Acc: {true_cv_accuracy:.3f})")
                        
                        # Save the results
                        xgb_experiment_results.append({
                            'Sample_Source': source,
                            'Target_Toxin': target,
                            'Split': i,
                            'Preprocessing': pipe_name,
                            'Feature_Method': fp_name,
                            'Best_n_estimators': random_search.best_params_['n_estimators'],
                            'Best_learning_rate': random_search.best_params_['learning_rate'],
                            'Best_max_depth': random_search.best_params_['max_depth'],
                            'Best_subsample': random_search.best_params_['subsample'],
                            'Best_colsample_bytree': random_search.best_params_['colsample_bytree'],
                            'Best_gamma': random_search.best_params_['gamma'],
                            'Best_Colsample_bylevel': random_search.best_params_['colsample_bylevel'],
                            'Best_min_child_weight': random_search.best_params_['min_child_weight'],
                            'Best_reg_alpha': random_search.best_params_['reg_alpha'],
                            'Best_reg_lambda': random_search.best_params_['reg_lambda'],
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

# Combine all results into a dataframe for easier analysis
optimizied_algorithm = pd.DataFrame(xgb_experiment_results)
optimizied_algorithm.to_csv('./results/global_with_country/2_XGB_results.csv', index=False)