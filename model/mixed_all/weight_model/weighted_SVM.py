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
from sklearn.svm import SVC

# Import custom configurations
from model.pipeline_config import (preprocessing_pipelines, feature_pipelines, 
                                   scoring_metrics, load_nirs_data)

# Import the weighted data
X_train, X_test, y_train_weighted_binary, y_test_weighted_binary, train_country, test_country = load_nirs_data(
    splits=10, threshold=1000, weighted=True
)

# Define the sources and targets for iteration
sources = ['intact', 'ground']
targets = ['DON', 'Total_DON']

# ==========================================
# SVM MODEL OPTIMIZATION
# ==========================================
wsvm_experiment_results = []

# Update parameter distributions to match the Pipeline naming convention 
svm_param_dist = {
    'svm__C': [0.1, 1, 5, 10, 50, 100, 500, 1000],
    'svm__gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
}

# Initialize the encoder for the countries
country_encoder = OrdinalEncoder()

for source in sources:
    for target in targets:
        for i in range(10): 
            print(f"\nEvaluating: {source} samples for {target} (Split {i+1}/10)")
            
            # Extract raw data arrays
            X_tr_spectra = X_train[source][i].values
            X_te_spectra = X_test[source][i].values
            
            # Explicitly map 'HC' to 1 and 'LC' to 0
            y_tr = np.where(y_train_weighted_binary[source][i][target].values == 'HC', 1, 0)
            y_te = np.where(y_test_weighted_binary[source][i][target].values == 'HC', 1, 0)

            # Extract sample weights
            tr_weights = y_train_weighted_binary[source][i][target + '_weight'].values
            te_weights = y_test_weighted_binary[source][i][target + '_weight'].values
            
            for pipe_name, preprocessor in preprocessing_pipelines.items():
                for fp_name, fp_method in feature_pipelines.items():
                    print(f"  -> Running Prep: {pipe_name} | Feature: {fp_name}...", end=" ")
                    start_time = time.time()
                    
                    try:
                        # --------------------------------------------------------
                        # 1. BUILD THE TUNING PIPELINE
                        # --------------------------------------------------------
                        steps = []
                        if preprocessor is not None:
                            steps.append(('prep', preprocessor))
                        if fp_method is not None:
                            steps.append(('feature_selection', fp_method))
                            
                        # Base SVC model
                        base_svm = SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=270725)
                        steps.append(('svm', base_svm))
                        
                        tuning_pipeline = Pipeline(steps)
                        
                        # --------------------------------------------------------
                        # INNER LOOP: HYPERPARAMETER TUNING (With Weights)
                        # --------------------------------------------------------
                        random_search = RandomizedSearchCV(
                            estimator=tuning_pipeline,
                            param_distributions=svm_param_dist,
                            n_iter=20,          
                            cv=10,
                            scoring=scoring_metrics,
                            refit='roc_auc',
                            n_jobs=-1,          
                            random_state=270725
                        )
                        
                        # Route the weights specifically to the SVM step inside the Pipeline
                        random_search.fit(X_tr_spectra, y_tr, svm__sample_weight=tr_weights)
                        
                        # --------------------------------------------------------
                        # REBUILD PIPELINE WITH INTERNAL CALIBRATOR
                        # --------------------------------------------------------
                        # Extract ONLY the winning SVM to wrap in the calibrator
                        best_svm = random_search.best_estimator_.named_steps['svm']
                        calibrated_svm = CalibratedClassifierCV(best_svm, cv=10, method='sigmoid')

                        # Build the final pipeline with the calibrator safely nestled inside
                        final_steps = []
                        if preprocessor is not None:
                            final_steps.append(('prep', preprocessor))
                        if fp_method is not None:
                            final_steps.append(('feature_selection', fp_method))
                        final_steps.append(('calibrated_svm', calibrated_svm))
                        
                        final_pipeline = Pipeline(final_steps)

                        # --------------------------------------------------------
                        # OUT-OF-FOLD CALIBRATION & THRESHOLD OPTIMIZATION
                        # --------------------------------------------------------
                        cv_strategy = StratifiedKFold(n_splits=10, shuffle=True, random_state=270725)
                        
                        # Safely route weights to the calibrator inside the final pipeline
                        calibrated_cv_probs = cross_val_predict(
                            final_pipeline, X_tr_spectra, y_tr, 
                            cv=cv_strategy, method='predict_proba', n_jobs=-1,
                            params={'calibrated_svm__sample_weight': tr_weights} 
                        )
                        
                        # Find optimal threshold using Youden's J (Weighted)
                        fpr, tpr, thresholds = roc_curve(y_tr, calibrated_cv_probs[:, 1], sample_weight=tr_weights)
                        best_idx = np.argmax(tpr - fpr)
                        optimal_threshold = thresholds[best_idx]
                        
                        calibrated_cv_preds = (calibrated_cv_probs[:, 1] >= optimal_threshold).astype(int)
                        
                        # Calculate true CV metrics (Weighted)
                        true_cv_auc = roc_auc_score(y_tr, calibrated_cv_probs[:, 1], sample_weight=tr_weights)
                        true_cv_accuracy = accuracy_score(y_tr, calibrated_cv_preds, sample_weight=tr_weights)
                        true_cv_sensitivity = recall_score(y_tr, calibrated_cv_preds, sample_weight=tr_weights)
                        true_cv_specificity = recall_score(y_tr, calibrated_cv_preds, pos_label=0, sample_weight=tr_weights)
                        
                        # --------------------------------------------------------
                        # OUTER LOOP: FINAL EXAMINATION ON TEST SET
                        # --------------------------------------------------------
                        # Fit the entire architecture from scratch on all training data
                        final_pipeline.fit(X_tr_spectra, y_tr, calibrated_svm__sample_weight=tr_weights)
                        
                        # Predict on the raw test vault!
                        test_probs = final_pipeline.predict_proba(X_te_spectra)[:, 1]
                        test_preds = (test_probs >= optimal_threshold).astype(int)

                        # Extract test metrics
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
                        # SAVE RESULTS
                        # --------------------------------------------------------
                        wsvm_experiment_results.append({
                            'Sample_Source': source,
                            'Target_Toxin': target,
                            'Preprocessing': pipe_name,
                            'Split': i,
                            'Feature_Method': fp_name,
                            'Best_C': random_search.best_params_['svm__C'],
                            'Best_Gamma': random_search.best_params_['svm__gamma'],
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

# Combine all results into a dataframe and save
optimizied_algorithm = pd.DataFrame(wsvm_experiment_results)
optimizied_algorithm.to_csv('./results/mixed_all/weighted_SVM_results.csv', index=False)
print("\nAll combinations processed successfully! Weighted SVM data saved to CSV.")