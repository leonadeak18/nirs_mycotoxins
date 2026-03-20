import pandas as pd
from sklearn.pipeline import Pipeline

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
y_train2, y_test2 = {}, {}
for source in sources:
    y_train2[source] = categorize_targets(y_train[source][['DON', 'Total_DON']])
    y_test2[source] = categorize_targets(y_test[source][['DON', 'Total_DON']])

#Preprocessing pipelines
import chemotools.scatter as ch
from chemotools.derivative import SavitzkyGolay
preprocessing_pipelines = {
    "Raw": None,
    "SNV_Smoothing": Pipeline([
        ('snv', ch.StandardNormalVariate()),
        ('smoothing', SavitzkyGolay(window_size=15, polynomial_order=2, derivate_order = 0))
    ]),
    "EMSC_Smoothing": Pipeline([
        ('emsc', ch.ExtendedMultiplicativeScatterCorrection()),
        ('smoothing', SavitzkyGolay(window_size=15, polynomial_order=2, derivate_order = 0))
    ]),
    "1st_derivative": Pipeline([
        ('first_derivative', SavitzkyGolay(window_size=15, polynomial_order=2, derivate_order = 1))
    ]),
    "2nd_derivative": Pipeline([
        ('second_derivative', SavitzkyGolay(window_size=15, polynomial_order=2, derivate_order = 2))
    ]),
    "RNV_Smoothing": Pipeline([
        ('rnv', ch.RobustNormalVariate()),
        ('smoothing', SavitzkyGolay(window_size=15, polynomial_order=2, derivate_order = 0))
    ])
}


## Apply preprocessing and plotting
X_train_processed, X_test_processed = {}, {}
for source in sources:
    X_train_processed[source], X_test_processed[source] = {}, {}
    for prep_name, prep_pipeline in preprocessing_pipelines.items():
        if prep_pipeline is not None:
            X_train_processed[source][prep_name] = pd.DataFrame(prep_pipeline.fit_transform(X_train[source]), index=X_train[source].index, columns=X_train[source].columns)
            X_test_processed[source][prep_name] = pd.DataFrame(prep_pipeline.transform(X_test[source]), index=X_test[source].index, columns=X_test[source].columns)
        else:
            X_train_processed[source][prep_name] = pd.DataFrame(X_train[source].values, index=X_train[source].index, columns=X_train[source].columns)
            X_test_processed[source][prep_name] = pd.DataFrame(X_test[source].values, index=X_test[source].index, columns=X_test[source].columns)
