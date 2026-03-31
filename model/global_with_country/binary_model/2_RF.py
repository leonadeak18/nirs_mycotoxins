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
from sklearn.ensemble import RandomForestClassifier

#import the data
X_train, X_test, y_train_binary, y_test_binary, train_country, test_country = load_nirs_data(splits=10, threshold=1000, weighted=False)

# define the sources and targets for iteration
sources = ['intact', 'ground']
targets = ['DON', 'Total_DON']


# ==========================================
# RF MODEL OPTIMIZATION
# ==========================================
# Define the grid of parameters for the Random Forest to test
rf_param_dist = {
    'n_estimators': [100, 200, 300, 500, 700, 1000], 
    'max_features': ['sqrt', 'log2', 0.2, 0.3, 0.5], 
    'max_depth': [None, 5, 10, 20, 30, 40, 50],              
    'min_samples_split': [2, 5, 10, 20, 50],
    'min_samples_leaf': [1, 2, 4, 10, 20, 40, 50]
}

# Initialize the encoder for the countries
country_encoder = OrdinalEncoder()
rf_experiment_results = []
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
            country_tr_raw = train_country[source][i].values.reshape(-1, 1)
            country_te_raw = test_country[source][i].values.reshape(-1, 1)
            
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

                        # Define base RF model
                        base_rf = RandomForestClassifier(class_weight='balanced', random_state=270725)
                        
                        # Run Randomized Search to find the best parameters on the COMBINED data
                        random_search = RandomizedSearchCV(
                            estimator=base_rf,
                            param_distributions=rf_param_dist,
                            n_iter=20,          
                            cv=10,
                            scoring=scoring_metrics,
                            refit='roc_auc',
                            n_jobs=-1,          
                            random_state=270725
                        )
                        random_search.fit(X_tr_final, y_tr)
                        
                        # Extract the winning RF and Calibrate it
                        best_rf = random_search.best_estimator_
                        best_rf.probability = True
                        calibrated_rf = CalibratedClassifierCV(best_rf, cv=10, method='sigmoid')

                        # Get true, out-of-fold calibrated probabilities for the training set
                        cv_strategy = StratifiedKFold(n_splits=10, shuffle=True, random_state=270725)
                        calibrated_cv_probs = cross_val_predict(
                            calibrated_rf, X_tr_final, y_tr, 
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
                        calibrated_rf.fit(X_tr_final, y_tr)
                        
                        # Test on the Test Set
                        test_probs = calibrated_rf.predict_proba(X_te_final)[:, 1]
                        
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
                        rf_experiment_results.append({
                            'Sample_Source': source,
                            'Target_Toxin': target,
                            'Split': i,
                            'Preprocessing': pipe_name,
                            'Feature_Method': fp_name,
                            'Best_n_estimators': random_search.best_params_['n_estimators'],
                            'Best_max_depth': random_search.best_params_['max_depth'],
                            'Best_max_features': random_search.best_params_['max_features'],
                            'Best_min_samples_split': random_search.best_params_['min_samples_split'],
                            'Best_min_samples_leaf': random_search.best_params_['min_samples_leaf'],
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
optimizied_algorithm = pd.DataFrame(rf_experiment_results)
optimizied_algorithm.to_csv('./results/global_with_country/2_RF_results.csv', index=False)