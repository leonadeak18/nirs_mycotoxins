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

# Assuming these are imported from your custom config file
from model.pipeline_config import (preprocessing_pipelines, feature_pipelines, 
                                   scoring_metrics, load_nirs_data)

# ========================================================
# DATA LOADING & INITIALIZATION
# ========================================================
# Import the data
X_train, X_test, y_train_binary, y_test_binary, train_country, test_country = load_nirs_data(
    splits=10, threshold=1000, weighted=False
)

# Define the sources and targets for iteration
sources = ['intact', 'ground']
targets = ['DON', 'Total_DON']

# Parameter distribution for RandomizedSearchCV 
svm_param_dist = {
    'svm__C': [0.1, 1, 5, 10, 50, 100, 500, 1000],
    'svm__gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
}

svm_experiment_results = []

# ========================================================
# MAIN NESTED CROSS-VALIDATION LOOP
# ========================================================
for source in sources:
    for target in targets:
        for i in range(10): 
            print(f"\nEvaluating: {source} samples for {target} (Split {i+1}/10)")
            
            # Extract the raw data arrays for this specific outer split
            X_tr_spectra = X_train[source][i].values
            X_te_spectra = X_test[source][i].values
            
            # Explicitly map 'HC' to 1 and 'LC' to 0
            y_tr = np.where(y_train_binary[source][i][target].values == 'HC', 1, 0)
            y_te = np.where(y_test_binary[source][i][target].values == 'HC', 1, 0)
            
            # Iterate through all your defined combinations
            for pipe_name, preprocessor in preprocessing_pipelines.items():
                for fp_name, fp_method in feature_pipelines.items():
                    print(f"  -> Running Prep: {pipe_name} | Feature: {fp_name}...", end=" ")
                    start_time = time.time()
                    
                    try:
                        # --------------------------------------------------------
                        # BUILD THE LEAK-PROOF PIPELINE
                        # --------------------------------------------------------
                        steps = []
                        
                        # Add preprocessor if it's not None
                        if preprocessor is not None:
                            steps.append(('prep', preprocessor))
                            
                        # Add feature selector if it's not None
                        if fp_method is not None:
                            steps.append(('feature_selection', fp_method))
                            
                        # Define and add the base SVC model
                        base_svm = SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=270725)
                        steps.append(('svm', base_svm))
                        
                        # Construct the final pipeline object
                        pipeline = Pipeline(steps)
                        
                        # --------------------------------------------------------
                        # INNER LOOP: HYPERPARAMETER TUNING
                        # --------------------------------------------------------
                        random_search = RandomizedSearchCV(
                            estimator=pipeline,
                            param_distributions=svm_param_dist,
                            n_iter=20,          
                            cv=10,
                            scoring=scoring_metrics,
                            refit='roc_auc',
                            n_jobs=-1,          
                            random_state=270725
                        )
                        
                        # Fit the search using the raw spectra. The pipeline handles preprocessing internally!
                        random_search.fit(X_tr_spectra, y_tr)
                        
                        # --------------------------------------------------------
                        # OUT-OF-FOLD CALIBRATION & THRESHOLD OPTIMIZATION
                        # --------------------------------------------------------
                        best_pipeline = random_search.best_estimator_
                        calibrated_pipeline = CalibratedClassifierCV(best_pipeline, cv=10, method='sigmoid')

                        # Generate honest, unbiased probabilities via cross_val_predict
                        cv_strategy = StratifiedKFold(n_splits=10, shuffle=True, random_state=270725)
                        calibrated_cv_probs = cross_val_predict(
                            calibrated_pipeline, X_tr_spectra, y_tr, 
                            cv=cv_strategy, method='predict_proba', n_jobs=-1
                        )
                        
                        # Find the optimal threshold using Youden's J statistic
                        fpr, tpr, thresholds = roc_curve(y_tr, calibrated_cv_probs[:, 1])
                        best_idx = np.argmax(tpr - fpr)
                        optimal_threshold = thresholds[best_idx]
                        
                        # Convert OOF probabilities to hard predictions
                        calibrated_cv_preds = (calibrated_cv_probs[:, 1] >= optimal_threshold).astype(int)
                        
                        # Calculate true CV metrics
                        true_cv_auc = roc_auc_score(y_tr, calibrated_cv_probs[:, 1])
                        true_cv_accuracy = accuracy_score(y_tr, calibrated_cv_preds)
                        true_cv_sensitivity = recall_score(y_tr, calibrated_cv_preds)
                        true_cv_specificity = recall_score(y_tr, calibrated_cv_preds, pos_label=0)
                        
                        # --------------------------------------------------------
                        # OUTER LOOP: FINAL EXAMINATION ON TEST SET
                        # --------------------------------------------------------
                        # Fit the calibrator and entire pipeline on the FULL training set
                        calibrated_pipeline.fit(X_tr_spectra, y_tr)
                        
                        # Predict on the strictly held-out test data
                        test_probs = calibrated_pipeline.predict_proba(X_te_spectra)[:, 1]
                        test_preds = (test_probs >= optimal_threshold).astype(int)

                        # Extract final test metrics
                        test_auc = roc_auc_score(y_te, test_probs)
                        test_accuracy = accuracy_score(y_te, test_preds)
                        test_sensitivity = recall_score(y_te, test_preds)
                        test_specificity = recall_score(y_te, test_preds, pos_label=0, zero_division=0)
                        test_PR_curve = average_precision_score(y_te, test_preds)
                        test_f1 = f1_score(y_te, test_preds, zero_division=0)
                        test_mcc = matthews_corrcoef(y_te, test_preds)
                        
                        elapsed_time = time.time() - start_time
                        print(f"Done! (Test AUC: {test_auc:.3f} | CV Acc: {true_cv_accuracy:.3f})")
                        
                        # --------------------------------------------------------
                        # SAVE RESULTS
                        # --------------------------------------------------------
                        svm_experiment_results.append({
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

# ========================================================
# EXPORT FINAL DATA
# ========================================================
optimizied_algorithm = pd.DataFrame(svm_experiment_results)
optimizied_algorithm.to_csv('./results/mixed_all/2_SVM_results.csv', index=False)
print("\nAll combinations processed successfully! Data saved to CSV.")