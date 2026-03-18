import pandas as pd
from sklearn.preprocessing import StandardScaler
from lazypredict.Supervised import LazyRegressor
from sklearn.pipeline import Pipeline
import nirs4all.operators.transforms as tr

def load_nir_subset(path):
    """Loads NIR data and filters for wavelengths between 1100 and 2500 nm."""
    df = pd.read_csv(path, index_col=0)
    df = df.loc[:, (df.columns.astype(float) >= 1100) & (df.columns.astype(float) <= 2500)]
    return df

# Define the scopes
sources = ['intact', 'ground']
targets = ['Log_DON', 'Log_Total_DON']

# Initialize storage dictionaries
X_train, y_train = {}, {}
X_test, y_test = {}, {}

# Loop through all sources to load the data efficiently
for source in sources:
    X_train[source] = load_nir_subset(f'./nirs/train_test_set/X_train_{source}.csv')
    X_test[source] = load_nir_subset(f'./nirs/train_test_set/X_test_{source}.csv')
    
    y_train[source] = pd.read_csv(f'./nirs/train_test_set/y_train_{source}.csv', index_col=0)
    y_test[source] = pd.read_csv(f'./nirs/train_test_set/y_test_{source}.csv', index_col=0)


# We set up a dictionary so you can easily add or remove combinations later!
preprocessing_pipelines = {
    "RNV + Scaler": Pipeline([
        ('rnv', tr.RobustStandardNormalVariate()),
        ('scaler', StandardScaler())
    ]),
    
    "RNV + Smoothing + Scaler": Pipeline([
        ('rnv', tr.RobustStandardNormalVariate()),
        ('sg', tr.SavitzkyGolay(window_length=15, polyorder=2)), 
        ('scaler', StandardScaler())
    ]),

    "RNV + Detrend + Scaler": Pipeline([
        ('rnv', tr.RobustStandardNormalVariate()),
        ('sg', tr.SavitzkyGolay(window_length=15, polyorder=2)), 
        ('detrend', tr.Detrend()),
        ('scaler', StandardScaler())
    ])
}

all_results = {}

for source in sources:
    for target in targets:
        dataset_name = f"{source} predicting {target}"
        print(f"\n{'='*60}")
        print(f"🌟 Evaluating: {dataset_name} 🌟")
        print(f"{'='*60}")
        
        # Extract the specific data arrays for this loop iteration
        X_tr = X_train[source].values
        X_te = X_test[source].values
        
        # Ensure target variables are 1D arrays for LazyPredict
        y_tr = y_train[source][target].values
        y_te = y_test[source][target].values
        
        for pipe_name, preprocessor in preprocessing_pipelines.items():
            print(f"\n--- Preprocessing: {pipe_name} ---")
            
            # Fit/transform on training data, ONLY transform on testing data
            X_train_processed = preprocessor.fit_transform(X_tr)
            X_test_processed = preprocessor.transform(X_te)
            
            # Initialize and run LazyRegressor
            reg = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
            models, predictions = reg.fit(X_train_processed, X_test_processed, y_tr, y_te)
            
            # Save results to our dictionary
            result_key = f"{dataset_name} | {pipe_name}"
            all_results[result_key] = models
            
            # Print the top 3 models so you can monitor the progress
            print(models.head(3))

print("\n🎉 All combinations have been successfully screened!")

import joblib
my_project_data = {
    'X_train': X_train,
    'X_test': X_test,
    'y_train': y_train,
    'y_test': y_test,
    'preprocessing_pipelines': preprocessing_pipelines,
    'lazy_predict_results': all_results
}
joblib.dump(my_project_data, "exports/lazy_predictor_results.joblib")
