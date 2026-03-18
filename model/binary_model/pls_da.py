import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import nirs4all.operators.transforms as tr
from sklearn.decomposition import PCA, FastICA, KernelPCA, FactorAnalysis

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

## Categorize DON into 2 classes based on quantiles
def categorize_targets(y):
    bins = [-float('inf'), 1000, float('inf')]
    labels = ['LC', 'HC']
    
    y_class = pd.DataFrame(index=y.index)
    
    for target in targets:
        y_class[target] = pd.cut(y[target], bins=bins, labels=labels)
    
    return y_class

for source in sources:
    y_train[source] = categorize_targets(y_train[source][['DON', 'Total_DON']])
    y_test[source] = categorize_targets(y_test[source][['DON', 'Total_DON']])

#Preprocessing pipelines
preprocessing_pipelines = {
    "Smoothing + SNV": Pipeline([
        ('smoothing', tr.SavitzkyGolaySmoothing(window_length=11, polyorder=2)),
        ('snv', tr.StandardNormalVariate())
    ])
}