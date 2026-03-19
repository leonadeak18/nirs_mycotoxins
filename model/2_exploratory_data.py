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

for source in sources:
    y_train[source] = categorize_targets(y_train[source][['DON', 'Total_DON']])
    y_test[source] = categorize_targets(y_test[source][['DON', 'Total_DON']])

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

## Plotting the preprocessed spectra
### create a long format dataframe for plotting
from utils.plotting import format_spectra, plot_spectra
formatted_train = {}
for material in X_train_processed.keys():
    formatted_train[material] = {}
    for prep_name in X_train_processed[material].keys():
        formatted_train[material][prep_name] = format_spectra(X_train_processed[material][prep_name], material)

### plot the spectra
spectra_plot = {}
for material in formatted_train.keys():
    spectra_plot[material] = {}
    for prep_name in formatted_train[material].keys():
        print(f"Plotting {material} - {prep_name}")
        spectra_plot[material][prep_name] = plot_spectra(formatted_train[material][prep_name])
print("All spectra plotted successfully.")

#### create a subplots for all the spectra
from plotly.subplots import make_subplots
spectra_grid = {}
for material, preps in spectra_plot.items():
    prep_names = list(preps.keys())
    
    # 1. Create the blank canvas (3x2 grid)
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=prep_names,
        horizontal_spacing=0.1,
        vertical_spacing=0.2
    )

    # 2. ADD THE SPECTRA (This is the missing step!)
    for i, prep_name in enumerate(prep_names):
        # Stop at 6 plots so we don't break the 3x2 grid
        if i >= 6: break 
        
        # Calculate the correct grid placement (rows 1-2, cols 1-3)
        row = (i // 3) + 1
        col = (i % 3) + 1
        
        # Grab the individual figure you generated in your previous step
        single_fig = preps[prep_name]
        
        # Loop through the lines in that individual plot and add them to the grid box
        for trace in single_fig.data:
            trace.showlegend = False # Keep things tidy
            fig.add_trace(trace, row=row, col=col)

    # 3. Apply the clean theme to the entire grid
    fig.update_layout(
        title_text=f"<b>Spectra Comparison for {material}</b>",
        template='simple_white', 
        showlegend=False,
        font=dict(family="Arial", size=12) 
    )
    
    # 4. Apply axis settings to ALL subplots
    fig.update_xaxes(
        title_text="Wavelength (nm)", 
        dtick=300
    ) 
    
    fig.update_yaxes(
        title_text="Reflectance (1/R)"
    )

    spectra_grid[material] = fig

# Display the plot
spectra_grid['ground'].show()
spectra_grid['intact'].show()

#export the plot as a high-resolution image
# spectra_grid["ground"].write_image("./figures/preprocessing/ground.png")
# spectra_grid["intact"].write_image("./figures/preprocessing/intact.png")

# Dimentionality reduction techniques
from sklearn.decomposition import PCA, FastICA, KernelPCA, FactorAnalysis
dr_pipelines = {
    "None": None,
    "PCA": PCA(n_components=0.99),
    "ICA": FastICA(n_components=0.99, random_state=123),
    "KernelPCA": KernelPCA(n_components=0.99, kernel='rbf', gamma=1),
    "FA": FactorAnalysis(n_components=0.99)
}
