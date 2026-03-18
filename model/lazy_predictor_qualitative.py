import pandas as pd
from sklearn.preprocessing import StandardScaler
from lazypredict.Supervised import LazyClassifier
from sklearn.pipeline import Pipeline
import nirs4all.operators.transforms as tr
from sklearn.decomposition import PCA, KernelPCA, NMF, FactorAnalysis

def load_nir_subset(path):
    """Loads NIR data and filters for wavelengths between 1100 and 2500 nm."""
    df = pd.read_csv(path, index_col=0)
    df = df.loc[:, (df.columns.astype(float) >= 1100) & (df.columns.astype(float) <= 2500)]
    return df

# Define the scopes
sources = ['intact', 'ground']
targets = ['DON', 'Total_DON']

# Initialize storage dictionaries
X_train, y_train = {}, {}
X_test, y_test = {}, {}

# Loop through all sources to load the data efficiently
for source in sources:
    X_train[source] = load_nir_subset(f'./nirs/train_test_set/X_train_{source}.csv')
    X_test[source] = load_nir_subset(f'./nirs/train_test_set/X_test_{source}.csv')
    
    y_train[source] = pd.read_csv(f'./nirs/train_test_set/y_train_{source}.csv', index_col=0)
    y_test[source] = pd.read_csv(f'./nirs/train_test_set/y_test_{source}.csv', index_col=0)

for key in y_test:
    #remove Total_DON and Log_Total_DON columns from y_test
    y_test[key] = y_test[key].drop(columns=['Log_DON', 'Log_Total_DON'])

for key in y_train:
    #remove Total_DON and Log_Total_DON columns from y_train
    y_train[key] = y_train[key].drop(columns=['Log_DON', 'Log_Total_DON'])

## Categorize DON into 3 classes based on quantiles
def categorize_targets(y):
    bins = [-float('inf'), 500, 1500, float('inf')]
    labels = ['LC', 'UC', 'HC']
    
    y_class = pd.DataFrame(index=y.index)
    
    for target in targets:
        y_class[target] = pd.cut(y[target], bins=bins, labels=labels)
    
    return y_class

for source in sources:
    y_train[source] = categorize_targets(y_train[source][['DON', 'Total_DON']])
    y_test[source] = categorize_targets(y_test[source][['DON', 'Total_DON']])


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

dim_reduction_methods = {
    "None": None,
    "PCA": PCA(n_components=20),
    "KernelPCA": KernelPCA(n_components=20, kernel='rbf', gamma=0.01),
    "FA": FactorAnalysis(n_components=20)
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
            # Fit/transform on training data, ONLY transform on testing data
                X_train_raw = preprocessor.fit_transform(X_tr)
                X_test_raw = preprocessor.transform(X_te)

                for dr_name, dr_method in dim_reduction_methods.items():
                    print(f"\n--- Preprocessing: {pipe_name} | Dimensionality Reduction: {dr_name} ---")
                    if dr_method is not None:
                        X_tr_final = dr_method.fit_transform(X_train_raw)
                        X_te_final = dr_method.transform(X_test_raw)
                    else:
                        X_tr_final = X_train_raw
                        X_te_final = X_test_raw
                
                    # Initialize and run LazyClassifier
                    clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
                    models, predictions = clf.fit(X_tr_final, X_te_final, y_tr, y_te)
                    
                    # Save results to our dictionary
                    result_key = f"{dataset_name} | {pipe_name} | {dr_name}"
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
joblib.dump(my_project_data, "exports/lazy_predictor_qualitative.joblib")
