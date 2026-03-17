<<<<<<< HEAD
## Load all the necessary libraries
import pandas as pd
import numpy as np

## import all training files
X_train = {
    'intact': pd.read_csv('./nirs/train_test_set/X_train_intact.csv', index_col=0),
    'ground': pd.read_csv('./nirs/train_test_set/X_train_ground.csv', index_col=0),
    'abs_ground': pd.read_csv('./nirs/train_test_set/X_train_abs_ground.csv', index_col=0)
}
y_train = {
    'intact': pd.read_csv('./nirs/train_test_set/y_train_intact.csv', index_col=0),
    'ground': pd.read_csv('./nirs/train_test_set/y_train_ground.csv', index_col=0),
    'abs_ground': pd.read_csv('./nirs/train_test_set/y_train_abs_ground.csv', index_col=0)
}
for key in y_train:
    #remove Total_DON and Log_Total_DON columns from y_train
    y_train[key] = y_train[key].drop(columns=['Total_DON', 'Log_Total_DON'])

X_test = {
    'intact': pd.read_csv('./nirs/train_test_set/X_test_intact.csv', index_col=0),
    'ground': pd.read_csv('./nirs/train_test_set/X_test_ground.csv', index_col=0),
    'abs_ground': pd.read_csv('./nirs/train_test_set/X_test_abs_ground.csv', index_col=0)
}
y_test = {
    'intact': pd.read_csv('./nirs/train_test_set/y_test_intact.csv', index_col=0),
    'ground': pd.read_csv('./nirs/train_test_set/y_test_ground.csv', index_col=0),
    'abs_ground': pd.read_csv('./nirs/train_test_set/y_test_abs_ground.csv', index_col=0)
}
for key in y_test:
    #remove Total_DON and Log_Total_DON columns from y_test
    y_test[key] = y_test[key].drop(columns=['Total_DON', 'Log_Total_DON'])

## preprocessing the spectral data
'''
here are the list of preprocessing steps that we will apply to the spectral data:
1. Scatter correction: SNV, MSC, EMSC, RNV
2. Reduce noisy spectra: Savitzky-Golay smoothing
3. Baseline Correction: Detrending, First and Second Derivatives
4. Wavelet transformation
5. Scaling
'''
import nirs4all
from nirs4all.operators import transforms as tr
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import StandardScaler, RobustScaler

pipeline =[
    # Scatter correction
    {
        "feature_augmentation":{
            "_or_": [
                None,
                tr.StandardNormalVariate(),
                tr.MultiplicativeScatterCorrection(),
                tr.ExtendedMultiplicativeScatterCorrection(degree=3),
                tr.RobustStandardNormalVariate()
            ], "pick":1},
            "action":"extend" ## to make a branch of preprocessing
        
    },
    # smoothing
    {
        "feature_augmentation": {
            "_or_": [
                None,
                tr.SavitzkyGolay(window_length=15, polyorder=2),
                tr.SavitzkyGolay(window_length=21, polyorder=2),
                tr.SavitzkyGolay(window_length=15, polyorder=3),
                tr.SavitzkyGolay(window_length=21, polyorder=3)
            ], "pick":1},
            "action": "add"
    },

    # baseline correction and Derivatives
    {
        "feature_augmentation":{
            "_or_": [
                None,
                tr.Detrend(),
                tr.FirstDerivative(delta = 0.5, edge_order=2), ## delta is the spacing between points, edge_order is the order of the polynomial used to fit the edges of the spectrum
                tr.SecondDerivative(delta = 0.5, edge_order=2)
            ],
            "pick": 1},
            "action": "add"
    },

    # Scaling
    {
        "feature_augmentation":{
            "_or_": [
                None,
                StandardScaler(),
                RobustScaler()
            ],
            "pick": 1},
            "action": "add"
    },
    ## shuffle split for cross validation
    {
        "cross_validation": ShuffleSplit(n_splits=10, random_state=123)
    },
    ## PLSR model
    {
        "model": PLSRegression(scale=False),
        "finetune_params":{
            "n_trials": 50,
            "sample": "tpe",
            "approach": "grouped",
            "model_params":{
                "n_components": ('int', 1, 20)
            }
        }
    }
]

# Create a list of dictionaries to define multiple sources
sources = ['intact', 'ground', 'abs_ground']
targets = ['DON', 'Log_DON']

multi_source_dataset = [
    {
        "source_name": source,
        "train_x": X_train[source].values, 
        "train_y": y_train[source][target].values,
        "test_x": X_test[source].values,
        "test_y": y_test[source][target].values
    }
    for source in sources
    for target in targets
]


plsr_result = nirs4all.run(
    pipeline = pipeline,
    dataset = multi_source_dataset,
    verbose = 1,
    name = "PLSR_Search"
)

## Export my results to joblib file
import joblib
plsr_result.export("exports/best_model.n4a")

print(f"\nTotal variants explored: {plsr_result.num_predictions}")
print(f"Best Score (MSE): {plsr_result.best_score:.4f}")
print(f"Best Preprocessing Chain: {plsr_result.best.get('preprocessings', 'N/A')}")


# Show all results ranked
print("\nAll preprocessing variants (ranked by R²):")
for i, pred in enumerate(plsr_result.top(20, display_metrics=['r2']), 1):
    print(f"   {i}. {pred.get('preprocessings', 'N/A')}: R²={pred.get('r2', 0):.4f}")

best = plsr_result.best
print("\n🏆 Best Preprocessing Chain:")
print(f"   Chain: {best.get('preprocessings', 'N/A')}")
print(f"   Model: {best.get('model_name', 'N/A')}")
print(f"   Dataset: {best.get('dataset_name', 'N/A')}")
print(f"   Fold: {best.get('fold_id', 'N/A')}")

# Get detailed metrics
print("\n📊 Performance Metrics:")
print(f"   Primary metric ({best.get('metric', 'mse')}):")
print(f"   • Train: {best.get('train_score', float('nan')):.6f}")
print(f"   • Val:   {best.get('val_score', float('nan')):.6f}")
print(f"   • Test:  {best.get('test_score', float('nan')):.6f}")

# Get additional metrics using display_metrics
top_with_metrics = plsr_result.top(n=20, display_metrics=['rmse', 'r2', 'mae'])
if top_with_metrics:
    best_with_metrics = top_with_metrics[1]
    print("\n   Additional metrics:")
    print(f"   • RMSE: {best_with_metrics.get('rmse', float('nan')):.4f}")
    print(f"   • R²:   {best_with_metrics.get('r2', float('nan')):.4f}")
    print(f"   • MAE:  {best_with_metrics.get('mae', float('nan')):.4f}")


# Get the top 5 performing models from your run
for dataset_name, dataset_info in plsr_result.per_dataset.items():
    dataset_predictions = dataset_info.get('run_predictions')
    
    if dataset_predictions:
        # Pull the absolute best model for this specific dataset
        best_model = dataset_predictions.top(n=1, rank_metric='r2')
        print(f"Dataset: {dataset_name} | Best Model: {best_model['model_name']} | R²: {best_model['r2']:.4f}")

all_results = plsr_result.top(
    n=plsr_result.num_predictions, 
    display_metrics=['mse', 'rmse', 'r2', 'mae']
)
df_results = pd.DataFrame(all_results)
df_results.to_csv("exports/all_plsr_metrics.csv", index=False)
=======
## Load all the necessary libraries
import pandas as pd
import numpy as np

## import all training files
X_train = {
    'intact': pd.read_csv('./nirs/train_test_set/X_train_intact.csv', index_col=0),
    'ground': pd.read_csv('./nirs/train_test_set/X_train_ground.csv', index_col=0),
    'abs_ground': pd.read_csv('./nirs/train_test_set/X_train_abs_ground.csv', index_col=0)
}
y_train = {
    'intact': pd.read_csv('./nirs/train_test_set/y_train_intact.csv', index_col=0),
    'ground': pd.read_csv('./nirs/train_test_set/y_train_ground.csv', index_col=0),
    'abs_ground': pd.read_csv('./nirs/train_test_set/y_train_abs_ground.csv', index_col=0)
}
for key in y_train:
    #remove Total_DON and Log_Total_DON columns from y_train
    y_train[key] = y_train[key].drop(columns=['Total_DON', 'Log_Total_DON'])

X_test = {
    'intact': pd.read_csv('./nirs/train_test_set/X_test_intact.csv', index_col=0),
    'ground': pd.read_csv('./nirs/train_test_set/X_test_ground.csv', index_col=0),
    'abs_ground': pd.read_csv('./nirs/train_test_set/X_test_abs_ground.csv', index_col=0)
}
y_test = {
    'intact': pd.read_csv('./nirs/train_test_set/y_test_intact.csv', index_col=0),
    'ground': pd.read_csv('./nirs/train_test_set/y_test_ground.csv', index_col=0),
    'abs_ground': pd.read_csv('./nirs/train_test_set/y_test_abs_ground.csv', index_col=0)
}
for key in y_test:
    #remove Total_DON and Log_Total_DON columns from y_test
    y_test[key] = y_test[key].drop(columns=['Total_DON', 'Log_Total_DON'])

## preprocessing the spectral data
'''
here are the list of preprocessing steps that we will apply to the spectral data:
1. Scatter correction: SNV, MSC, EMSC, RNV
2. Reduce noisy spectra: Savitzky-Golay smoothing
3. Baseline Correction: Detrending, First and Second Derivatives
4. Wavelet transformation
5. Scaling
'''
import nirs4all
from nirs4all.operators import transforms as tr
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import StandardScaler, RobustScaler

pipeline =[
    # Scatter correction
    {
        "feature_augmentation":{
            "_or_": [
                None,
                tr.StandardNormalVariate(),
                tr.MultiplicativeScatterCorrection(),
                tr.ExtendedMultiplicativeScatterCorrection(degree=3),
                tr.RobustStandardNormalVariate()
            ], "pick":1},
            "action":"extend" ## to make a branch of preprocessing
        
    },
    # smoothing
    {
        "feature_augmentation": {
            "_or_": [
                None,
                tr.SavitzkyGolay(window_length=15, polyorder=2),
                tr.SavitzkyGolay(window_length=21, polyorder=2),
                tr.SavitzkyGolay(window_length=15, polyorder=3),
                tr.SavitzkyGolay(window_length=21, polyorder=3)
            ], "pick":1},
            "action": "add"
    },

    # baseline correction and Derivatives
    {
        "feature_augmentation":{
            "_or_": [
                None,
                tr.Detrend(),
                tr.FirstDerivative(delta = 0.5, edge_order=2), ## delta is the spacing between points, edge_order is the order of the polynomial used to fit the edges of the spectrum
                tr.SecondDerivative(delta = 0.5, edge_order=2)
            ],
            "pick": 1},
            "action": "add"
    },

    # Scaling
    {
        "feature_augmentation":{
            "_or_": [
                None,
                StandardScaler(),
                RobustScaler()
            ],
            "pick": 1},
            "action": "add"
    },
    ## shuffle split for cross validation
    {
        "cross_validation": ShuffleSplit(n_splits=10, random_state=123)
    },
    ## PLSR model
    {
        "model": PLSRegression(scale=False),
        "finetune_params":{
            "n_trials": 50,
            "sample": "tpe",
            "approach": "grouped",
            "model_params":{
                "n_components": ('int', 1, 20)
            }
        }
    }
]

# Create a list of dictionaries to define multiple sources
sources = ['intact', 'ground', 'abs_ground']
targets = ['DON', 'Log_DON']

multi_source_dataset = [
    {
        "source_name": source,
        "train_x": X_train[source].values, 
        "train_y": y_train[source][target].values,
        "test_x": X_test[source].values,
        "test_y": y_test[source][target].values
    }
    for source in sources
    for target in targets
]


plsr_result = nirs4all.run(
    pipeline = pipeline,
    dataset = multi_source_dataset,
    verbose = 1,
    name = "PLSR_Search"
)

## Export my results to joblib file
import joblib
plsr_result.export("exports/best_model.n4a")

print(f"\nTotal variants explored: {plsr_result.num_predictions}")
print(f"Best Score (MSE): {plsr_result.best_score:.4f}")
print(f"Best Preprocessing Chain: {plsr_result.best.get('preprocessings', 'N/A')}")


# Show all results ranked
print("\nAll preprocessing variants (ranked by R²):")
for i, pred in enumerate(plsr_result.top(20, display_metrics=['r2']), 1):
    print(f"   {i}. {pred.get('preprocessings', 'N/A')}: R²={pred.get('r2', 0):.4f}")

best = plsr_result.best
print("\n🏆 Best Preprocessing Chain:")
print(f"   Chain: {best.get('preprocessings', 'N/A')}")
print(f"   Model: {best.get('model_name', 'N/A')}")
print(f"   Dataset: {best.get('dataset_name', 'N/A')}")
print(f"   Fold: {best.get('fold_id', 'N/A')}")

# Get detailed metrics
print("\n📊 Performance Metrics:")
print(f"   Primary metric ({best.get('metric', 'mse')}):")
print(f"   • Train: {best.get('train_score', float('nan')):.6f}")
print(f"   • Val:   {best.get('val_score', float('nan')):.6f}")
print(f"   • Test:  {best.get('test_score', float('nan')):.6f}")

# Get additional metrics using display_metrics
top_with_metrics = plsr_result.top(n=20, display_metrics=['rmse', 'r2', 'mae'])
if top_with_metrics:
    best_with_metrics = top_with_metrics[1]
    print("\n   Additional metrics:")
    print(f"   • RMSE: {best_with_metrics.get('rmse', float('nan')):.4f}")
    print(f"   • R²:   {best_with_metrics.get('r2', float('nan')):.4f}")
    print(f"   • MAE:  {best_with_metrics.get('mae', float('nan')):.4f}")


# Get the top 5 performing models from your run
for dataset_name, dataset_info in plsr_result.per_dataset.items():
    dataset_predictions = dataset_info.get('run_predictions')
    
    if dataset_predictions:
        # Pull the absolute best model for this specific dataset
        best_model = dataset_predictions.top(n=1, rank_metric='r2')
        print(f"Dataset: {dataset_name} | Best Model: {best_model['model_name']} | R²: {best_model['r2']:.4f}")

all_results = plsr_result.top(
    n=plsr_result.num_predictions, 
    display_metrics=['mse', 'rmse', 'r2', 'mae']
)
df_results = pd.DataFrame(all_results)
df_results.to_csv("exports/all_plsr_metrics.csv", index=False)
>>>>>>> c0b013e83447d4824ced8d39df8883c0ce891429
