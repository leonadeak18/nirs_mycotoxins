from plotnine import scale_y_continuous
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



from utils.preparation import binary_transform, three_level_transform
# categorize into two classes: Low Contamination (LC) and High Contamination (HC)
y_train2, y_test2 = {}, {}
for source in sources:
    y_train2[source] = binary_transform(y_train[source][['DON', 'Total_DON']])
    y_test2[source] = binary_transform(y_test[source][['DON', 'Total_DON']])

#Preprocessing pipelines
import chemotools.scatter as ch
from chemotools.derivative import SavitzkyGolay
preprocessing_pipelines = {
    "Raw": None,
    "SNV_Smoothing": Pipeline([
        ('snv', ch.StandardNormalVariate()),
        ('smoothing', SavitzkyGolay(window_length=15, polyorder=2, deriv = 0))
    ]),
    "EMSC_Smoothing": Pipeline([
        ('emsc', ch.ExtendedMultiplicativeScatterCorrection()),
        ('smoothing', SavitzkyGolay(window_length=15, polyorder=2, deriv = 0))
    ]),
    "1st_derivative": Pipeline([
        ('first_derivative', SavitzkyGolay(window_length=15, polyorder=2, deriv = 1))
    ]),
    "2nd_derivative": Pipeline([
        ('second_derivative', SavitzkyGolay(window_length=15, polyorder=2, deriv = 2))
    ]),
    "RNV_Smoothing": Pipeline([
        ('rnv', ch.RobustNormalVariate()),
        ('smoothing', SavitzkyGolay(window_length=15, polyorder=2, deriv = 0))
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
        spectra_plot[material][prep_name] = plot_spectra(formatted_train[material][prep_name], title = f"{prep_name}")
print("All spectra plotted successfully.")
spectra_plot['intact']['1st_derivative'].show()

#### create a grid of plots for each material
from plotnine import *
grid_spectra = {}
for material in spectra_plot.keys():
    grid_spectra[material] = (spectra_plot[material]['Raw']|spectra_plot[material]['SNV_Smoothing']|spectra_plot[material]['EMSC_Smoothing'])/(spectra_plot[material]['1st_derivative']|spectra_plot[material]['2nd_derivative']|spectra_plot[material]['RNV_Smoothing'])
    grid_spectra[material] = grid_spectra[material] + theme(figure_size=(24, 10))
    grid_spectra[material].save(f"./figures/preprocessing/all_spectra_grid_{material}.svg", dpi=600)
print("All grid plots saved successfully.")


## Plotting the spectra with the country as color
country = pd.read_csv('./nirs/location.csv', index_col=0)
country['country'] = country['country'].astype('category')
country['year'] = country['year'].astype('category')

### create a long format dataframe for plotting with the country as color
from plotnine import theme

# Format the data and merge targets
formatted_train_with_country = {}
for material, preps in X_train_processed.items():
    formatted_train_with_country[material] = {
        prep_name: format_spectra(data, material).merge(country, left_on='Sample_ID', right_index=True)
        for prep_name, data in preps.items()
    }

# Plot and save grids for both targets in one go
spectra_plots_country = {}
for material, preps in formatted_train_with_country.items():
    spectra_plots_country[material] = {}
    
    # A. Generate individual plots
    for prep_name, df in preps.items():
        print(f"Plotting {material} - {prep_name} with {target} target variable")
        spectra_plots_country[material][prep_name] = plot_spectra(
            df, 
            title=f"{prep_name}", 
            color_var='country'
        )
    
    # B. Assemble and save the grid for this material & target
    p = spectra_plots_country[material]
    grid = (p['Raw'] | p['SNV_Smoothing'] | p['EMSC_Smoothing']) / \
            (p['1st_derivative'] | p['2nd_derivative'] | p['RNV_Smoothing'])
    
    grid = grid + theme(figure_size=(24, 10))
    grid.save(f"./figures/preprocessing/country_all_spectra_grid_{material}.svg", dpi=600)

spectra_plots_year = {}
for material, preps in formatted_train_with_country.items():
    spectra_plots_year[material] = {}
    
    # A. Generate individual plots
    for prep_name, df in preps.items():
        print(f"Plotting {material} - {prep_name} with year as color")
        spectra_plots_year[material][prep_name] = plot_spectra(
            df, 
            title=f"{prep_name}", 
            color_var='year'
        ) + theme(legend_position="right")
    
    # B. Assemble and save the grid for this material & target
    p = spectra_plots_year[material]
    grid = (p['Raw'] | p['SNV_Smoothing'] | p['EMSC_Smoothing']) / \
            (p['1st_derivative'] | p['2nd_derivative'] | p['RNV_Smoothing'])
    grid = grid + theme(figure_size=(24, 10))
    grid.save(f"./figures/preprocessing/year_all_spectra_grid_{material}.svg", dpi=600)

## Plotting the spectra with the target variable as color
### create a long format dataframe for plotting with the target variable as color
from plotnine import theme

# Format the data and merge targets
formatted_train_with_target = {}
for material, preps in X_train_processed.items():
    formatted_train_with_target[material] = {
        prep_name: format_spectra(data, material).merge(y_train2[material], left_on='Sample_ID', right_index=True)
        for prep_name, data in preps.items()
    }

# Plot and save grids for both targets in one go
spectra_plots = {target: {} for target in targets}

for target in targets:
    for material, preps in formatted_train_with_target.items():
        spectra_plots[target][material] = {}
        
        # A. Generate individual plots
        for prep_name, df in preps.items():
            print(f"Plotting {material} - {prep_name} with {target} target variable")
            spectra_plots[target][material][prep_name] = plot_spectra(
                df, 
                title=f"{prep_name} with {target} categorization", 
                color_var=target
            )
        
        # B. Assemble and save the grid for this material & target
        p = spectra_plots[target][material]
        grid = (p['Raw'] | p['SNV_Smoothing'] | p['EMSC_Smoothing']) / \
               (p['1st_derivative'] | p['2nd_derivative'] | p['RNV_Smoothing'])
        
        grid = grid + theme(figure_size=(24, 10))
        grid.save(f"./figures/preprocessing/all_spectra_grid_{material}_{target}.svg", dpi=600)

# 3. Show your specific test plot
spectra_plots['DON']['intact']['SNV_Smoothing'].show()


## Categorize DON into 3 classes based on quantiles
y_train3, y_test3 = {}, {}
for source in sources:
    y_train3[source] = three_level_transform(y_train[source][['DON', 'Total_DON']])
    y_test3[source] = three_level_transform(y_test[source][['DON', 'Total_DON']])

# Format the data and merge targets
formatted_train_with_target3 = {}
for material, preps in X_train_processed.items():
    formatted_train_with_target3[material] = {
        prep_name: format_spectra(data, material).merge(y_train3[material], left_on='Sample_ID', right_index=True)
        for prep_name, data in preps.items()
    }

# Plot and save grids for both targets in one go
spectra_plots3 = {target: {} for target in targets}

for target in targets:
    for material, preps in formatted_train_with_target3.items():
        spectra_plots3[target][material] = {}
        
        # A. Generate individual plots
        for prep_name, df in preps.items():
            print(f"Plotting {material} - {prep_name} with {target} target variable")
            spectra_plots3[target][material][prep_name] = plot_spectra(
                df, 
                title=f"{prep_name} with {target} categorization", 
                color_var=target
            )
        
        # B. Assemble and save the grid for this material & target
        p = spectra_plots3[target][material]
        grid = (p['Raw'] | p['SNV_Smoothing'] | p['EMSC_Smoothing']) / \
               (p['1st_derivative'] | p['2nd_derivative'] | p['RNV_Smoothing'])
        
        grid = grid + theme(figure_size=(24, 10))
        grid.save(f"./figures/preprocessing/all_spectra_grid3_{material}_{target}.svg", dpi=600)


### Create a bar chart for the distribution of the categorized target variable
target_distribution = {}
for source in sources:
    target_distribution[source] = {}
    for target in targets:
        target_distribution[source][target] = y_train2[source][target].value_counts().reset_index()
        target_distribution[source][target].columns = ['Category', 'Count']
        target_distribution[source][target]['Source'] = source
        target_distribution[source][target]['Target'] = target
target_distribution_df = pd.concat([df for source in target_distribution.values() for df in source.values()], ignore_index=True)

bar_plot = (
    ggplot(target_distribution_df, aes(x='Source', y='Count', fill='Category')) +
    geom_bar(stat='identity', position='dodge') +
    facet_wrap('~Target') +
    labs(title='Distribution of Categorized Target Variables', x='Source', y='Count') +
    theme_classic() +
    scale_y_continuous(limits=(0, 250)) +
    scale_fill_manual(values=['#70c8e2', '#e54b2c']) +
    theme(
        title=element_text(size=12),
        axis_title=element_text(size=10),
        legend_position="right"
    )
)
bar_plot.save("./figures/preprocessing/target_distribution_bar_plot.svg", dpi=600)

## barchart for the test set
target_distribution_test = {}
for source in sources:
    target_distribution_test[source] = {}
    for target in targets:
        target_distribution_test[source][target] = y_test2[source][target].value_counts().reset_index()
        target_distribution_test[source][target].columns = ['Category', 'Count']
        target_distribution_test[source][target]['Source'] = source
        target_distribution_test[source][target]['Target'] = target
target_distribution_test_df = pd.concat([df for source in target_distribution_test.values() for df in source.values()], ignore_index=True)

bar_plot_test = (
    ggplot(target_distribution_test_df, aes(x='Source', y='Count', fill='Category')) +
    geom_bar(stat='identity', position='dodge') +
    facet_wrap('~Target') +
    labs(title='Distribution of Categorized Target Variables (Test Set)', x='Source', y='Count') +
    theme_classic() +
    scale_y_continuous(limits=(0, 70)) +
    scale_fill_manual(values=['#70c8e2', '#e54b2c']) +
    theme(
        title=element_text(size=12),
        axis_title=element_text(size=10),
        legend_position="right"
    )
)
bar_plot_test.save("./figures/preprocessing/target_distribution_bar_plot_test.svg", dpi=600)

### Bar chart for 3-category distribution
target_distribution3 = {}
for source in sources:
    target_distribution3[source] = {}
    for target in targets:
        target_distribution3[source][target] = y_train3[source][target].value_counts().reset_index()
        target_distribution3[source][target].columns = ['Category', 'Count']
        target_distribution3[source][target]['Source'] = source
        target_distribution3[source][target]['Target'] = target
target_distribution3_df = pd.concat([df for source in target_distribution3.values() for df in source.values()], ignore_index=True)

bar_plot3 = (
    ggplot(target_distribution3_df, aes(x='Source', y='Count', fill='Category')) +
    geom_bar(stat='identity', position='dodge') +
    facet_wrap('~Target') +
    labs(title='Distribution of 3-Category Target Variables', x='Source', y='Count') +
    theme_classic() +
    scale_y_continuous(limits=(0, 250)) +
    scale_fill_manual(values=['#70c8e2', 'black', '#e54b2c']) +
    theme(
        title=element_text(size=12),
        axis_title=element_text(size=10),
        legend_position="right"
    )
)
bar_plot3.save("./figures/preprocessing/target_distribution_bar_plot3.svg", dpi=600)


### Bar chart for 3-category distribution for test set
from plotnine import *
target_distribution3_test = {}
for source in sources:
    target_distribution3_test[source] = {}
    for target in targets:
        target_distribution3_test[source][target] = y_test3[source][target].value_counts().reset_index()
        target_distribution3_test[source][target].columns = ['Category', 'Count']
        target_distribution3_test[source][target]['Source'] = source
        target_distribution3_test[source][target]['Target'] = target
target_distribution3_test_df = pd.concat([df for source in target_distribution3_test.values() for df in source.values()], ignore_index=True)

bar_plot3_test = (
    ggplot(target_distribution3_test_df, aes(x='Source', y='Count', fill='Category')) +
    geom_bar(stat='identity', position='dodge') +
    facet_wrap('~Target') +
    labs(title='Distribution of 3-Category Target Variables (Test Set)', x='Source', y='Count') +
    theme_classic() +
    scale_y_continuous(limits=(0, 50)) +
    scale_fill_manual(values=['#70c8e2', 'black', '#e54b2c']) +
    theme(
        title=element_text(size=12),
        axis_title=element_text(size=10),
        legend_position="right"
    )
)
bar_plot3_test.save("./figures/preprocessing/target_distribution_bar_plot3_test.svg", dpi=600)

######## PCA analysis of the train spectra ############
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

pca_df = {}
pca_df2 = {}
pca_df3 = {}
pca_country = {}
pca_df_variance_explained = {}
for material in X_train_processed.keys():
    pca_df[material] = {}
    pca_df2[material] = {}
    pca_df3[material] = {}
    pca_country[material] = {}
    pca_df_variance_explained[material] = {}
    for prep_name in X_train_processed[material].keys():
        print(f"Performing PCA for {material} - {prep_name}")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train_processed[material][prep_name])
        
        pca = PCA(n_components=4)
        X_pca = pca.fit_transform(X_scaled)
        explained_variance = pca.explained_variance_ratio_
        pca_df_variance_explained[material][prep_name] = explained_variance
        
        pca_df[material][prep_name] = pd.DataFrame(X_pca, columns=['PC1', 'PC2', 'PC3', 'PC4'], index=X_train_processed[material][prep_name].index)
        pca_df2[material][prep_name] = pca_df[material][prep_name].merge(y_train2[material], left_index=True, right_index=True)
        pca_df3[material][prep_name] = pca_df[material][prep_name].merge(y_train3[material], left_index=True, right_index=True)
        pca_country[material][prep_name] = pca_df[material][prep_name].merge(country, left_index=True, right_index=True)

from utils.plotting import plot_pca
pca_plots = {}
for target in targets:
    pca_plots[target] = {}
    for material in pca_df2.keys():
        pca_plots[target][material] = {}
        for prep_name, df in pca_df2[material].items():
            print(f"Plotting PCA for {material} - {prep_name}")
            pca_plots[target][material][prep_name] = plot_pca(df, variance_explained=pca_df_variance_explained[material][prep_name], title=f"PCA of {prep_name}", color_var=target)
        
        #assemble and save the grid for this material & target
        p = pca_plots[target][material]
        grid = (p['Raw'] | p['SNV_Smoothing'] | p['EMSC_Smoothing']) / \
               (p['1st_derivative'] | p['2nd_derivative'] | p['RNV_Smoothing'])
        grid = grid + theme(figure_size=(24, 10))
        grid.save(f"./figures/PCA/pca_grid_{material}_{target}.svg", dpi=600)

    for material in pca_df3.keys():
        pca_plots[target][material] = {}
        for prep_name, df in pca_df3[material].items():
            print(f"Plotting PCA for {material} - {prep_name} with 3-category target")
            pca_plots[target][material][prep_name] = plot_pca(df, variance_explained=pca_df_variance_explained[material][prep_name], title=f"PCA of {prep_name} with 3-category target", color_var=target)
        
        #assemble and save the grid for this material & target
        p = pca_plots[target][material]
        grid = (p['Raw'] | p['SNV_Smoothing'] | p['EMSC_Smoothing']) / \
               (p['1st_derivative'] | p['2nd_derivative'] | p['RNV_Smoothing'])
        grid = grid + theme(figure_size=(24, 10))
        grid.save(f"./figures/PCA/pca_grid_{material}_{target}_3_category.svg", dpi=600)

#plotting based on country
pca_country_plot = {}
for material in pca_country.keys():
    pca_country_plot[material] = {}
    for prep_name, df in pca_country[material].items():
        print(f"Plotting PCA for {material} - {prep_name} with country categorization")
        pca_country_plot[material][prep_name] = plot_pca(df, variance_explained=pca_df_variance_explained[material][prep_name], title=f"PCA of {prep_name} with country categorization", color_var='country')
    
    #assemble and save the grid for this material & target
    p = pca_country_plot[material]
    grid = (p['Raw'] | p['SNV_Smoothing'] | p['EMSC_Smoothing']) / \
            (p['1st_derivative'] | p['2nd_derivative'] | p['RNV_Smoothing'])
    grid = grid + theme(figure_size=(24, 10))
    grid.save(f"./figures/PCA/country_pca_grid_{material}_category.svg", dpi=600)

pca_year = {}
for material in pca_country.keys():
    pca_year[material] = {}
    for prep_name, df in pca_country[material].items():
        print(f"Plotting PCA for {material} - {prep_name} with year categorization")
        pca_year[material][prep_name] = plot_pca(df, variance_explained=pca_df_variance_explained[material][prep_name], title=f"PCA of {prep_name} with year categorization", color_var='year') + theme(legend_position="right")
    
    #assemble and save the grid for this material & target
    p = pca_year[material]
    grid = (p['Raw'] | p['SNV_Smoothing'] | p['EMSC_Smoothing']) / \
            (p['1st_derivative'] | p['2nd_derivative'] | p['RNV_Smoothing'])
    grid = grid + theme(figure_size=(24, 10))
    grid.save(f"./figures/PCA/year_pca_grid_{material}_category.svg", dpi=600)
