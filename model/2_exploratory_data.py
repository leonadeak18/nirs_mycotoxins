########################################################################
##### Exploratory Data Analysis and Preprocessing of NIR Spectra  ######
########################################################################


####################################################
# Load the Data
####################################################

import pandas as pd
# Define the scopes
sources = ['intact', 'ground']
targets = ['DON', 'Total_DON']

# Loop through all sources to load the data efficiently
X, y = {}, {}
for source in sources:
    X[source] = pd.read_csv(
        f'./nirs/outlier_removal/nirs_{source}_cleaned.csv', 
        index_col=0, 
        dtype={0: str}
    )
    
    y[source] = pd.read_csv(
        f'./nirs/outlier_removal/y_{source}_cleaned.csv', 
        index_col=0, 
        dtype={0: str}
    )

from utils.preparation import binary_transform, three_level_transform
# categorize into two classes: Low Contamination (LC) and High Contamination (HC)
y2 = {}
for source in sources:
    y2[source] = {}
    y2[source] = binary_transform(y[source][['DON', 'Total_DON']])

y3 = {}
for source in sources:
    y3[source] = {}
    y3[source] = three_level_transform(y[source][['DON','Total_DON']])


####################################################
# Preprocessing and plotting the spectra
####################################################
## Pipelines
import chemotools.scatter as ch
from sklearn.pipeline import Pipeline
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
X_processed= {}
for source in sources:
    X_processed[source] = {}
    for prep_name, prep_pipeline in preprocessing_pipelines.items():
        if prep_pipeline is not None:
            X_processed[source][prep_name] = pd.DataFrame(prep_pipeline.fit_transform(X[source]), index=X[source].index, columns=X[source].columns)
        else:
            X_processed[source][prep_name] = pd.DataFrame(X[source].values, index=X[source].index, columns=X[source].columns)

## Plotting the preprocessed spectra
### create a long format dataframe for plotting
from utils.plotting import format_spectra, plot_spectra
formatted_X = {}
for material in X_processed.keys():
    formatted_X[material] = {}
    for prep_name in X_processed[material].keys():
        formatted_X[material][prep_name] = format_spectra(X_processed[material][prep_name], material)
        
### plot the spectra
spectra_plot = {}
for material in formatted_X.keys():
    spectra_plot[material] = {}
    for prep_name in formatted_X[material].keys():
        print(f"Plotting {material} - {prep_name}")
        spectra_plot[material][prep_name] = plot_spectra(formatted_X[material][prep_name], title = f"{prep_name}")
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
country = {}
for material in y:
    country[material] = y[material][['Country', 'Year']].astype('category')

### create a long format dataframe for plotting with the country as color
from plotnine import theme

formatted_train_with_country = {}

for material, preps in X_processed.items():
    formatted_train_with_country[material] = {}
    
    # Pre-select metadata
    meta = country[material][['Country', 'Year']]
    
    for prep_name, data in preps.items():
        # 1. This returns LONG format with a 'Sample_ID' column
        long_spectra = format_spectra(data, material)
        
        # 2. Merge 'Sample_ID' column (left) with country index (right)
        combined = long_spectra.merge(
            meta, 
            left_on='Sample_ID', 
            right_index=True, 
            how='inner'
        )
        
        formatted_train_with_country[material][prep_name] = combined

# Plot and save grids for both targets in one go
spectra_plots_country = {}
for material, preps in formatted_train_with_country.items():
    spectra_plots_country[material] = {}
    
    # A. Generate individual plots
    for prep_name, df in preps.items():
        print(f"Plotting {material} - {prep_name} with country")
        spectra_plots_country[material][prep_name] = plot_spectra(
            df, 
            title=f"{prep_name}", 
            color_var='Country'
        )
    
    # B. Assemble and save the grid for this material & target
    p = spectra_plots_country[material]
    grid = (p['Raw'] | p['SNV_Smoothing'] | p['EMSC_Smoothing']) / \
            (p['1st_derivative'] | p['2nd_derivative'] | p['RNV_Smoothing'])
    
    grid = grid + theme(figure_size=(24, 10))
    grid.save(f"./figures/preprocessing/country_all_spectra_grid_{material}.svg", dpi=600)

formatted_train_with_country['intact']['Raw']['Year'].nunique()

spectra_plots_year = {}
for material, preps in formatted_train_with_country.items():
    spectra_plots_year[material] = {}
    
    # A. Generate individual plots
    for prep_name, df in preps.items():
        print(f"Plotting {material} - {prep_name} with year as color")
        spectra_plots_year[material][prep_name] = plot_spectra(
            df, 
            title=f"{prep_name}", 
            color_var='Year'
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
for material, preps in X_processed.items():
    formatted_train_with_target[material] = {
        prep_name: format_spectra(data, material).merge(y2[material], left_on='Sample_ID', right_index=True)
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


# Format the data and merge targets
formatted_train_with_target3 = {}
for material, preps in X_processed.items():
    formatted_train_with_target3[material] = {
        prep_name: format_spectra(data, material).merge(y3[material], left_on='Sample_ID', right_index=True)
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
        target_distribution[source][target] = y2[source][target].value_counts().reset_index()
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
    scale_y_continuous(limits=(0, 350)) +
    scale_fill_manual(values=['#70c8e2', '#e54b2c']) +
    theme(
        title=element_text(size=12),
        axis_title=element_text(size=10),
        legend_position="right"
    )
)
bar_plot.save("./figures/preprocessing/target_distribution_bar_plot.svg", dpi=600)

### Bar chart for 3-category distribution
target_distribution3 = {}
for source in sources:
    target_distribution3[source] = {}
    for target in targets:
        target_distribution3[source][target] = y3[source][target].value_counts().reset_index()
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
    scale_y_continuous(limits=(0, 350)) +
    scale_fill_manual(values=['#70c8e2', 'black', '#e54b2c']) +
    theme(
        title=element_text(size=12),
        axis_title=element_text(size=10),
        legend_position="right"
    )
)
bar_plot3.save("./figures/preprocessing/target_distribution_bar_plot3.svg", dpi=600)



##################################################
######## PCA analysis of the  spectra ############
##################################################

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

pca_df = {}
pca_df2 = {}
pca_df3 = {}
pca_country = {}
pca_df_variance_explained = {}
for material in X_processed.keys():
    pca_df[material] = {}
    pca_df2[material] = {}
    pca_df3[material] = {}
    pca_country[material] = {}
    pca_df_variance_explained[material] = {}
    for prep_name in X_processed[material].keys():
        print(f"Performing PCA for {material} - {prep_name}")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_processed[material][prep_name])
        
        pca = PCA(n_components=4)
        X_pca = pca.fit_transform(X_scaled)
        explained_variance = pca.explained_variance_ratio_
        pca_df_variance_explained[material][prep_name] = explained_variance
        
        pca_df[material][prep_name] = pd.DataFrame(X_pca, columns=['PC1', 'PC2', 'PC3', 'PC4'], index=X_processed[material][prep_name].index)
        pca_df2[material][prep_name] = pca_df[material][prep_name].merge(y2[material], left_index=True, right_index=True)
        pca_df3[material][prep_name] = pca_df[material][prep_name].merge(y3[material], left_index=True, right_index=True)
        pca_country[material][prep_name] = pca_df[material][prep_name].merge(country[material], left_on = 'Sample_ID', right_index=True, how = 'inner')


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
        pca_country_plot[material][prep_name] = plot_pca(df, variance_explained=pca_df_variance_explained[material][prep_name], title=f"PCA of {prep_name} with country categorization", color_var='Country')
    
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
        pca_year[material][prep_name] = plot_pca(df, variance_explained=pca_df_variance_explained[material][prep_name], title=f"PCA of {prep_name} with year categorization", color_var='Year') + theme(legend_position="right")
    
    #assemble and save the grid for this material & target
    p = pca_year[material]
    grid = (p['Raw'] | p['SNV_Smoothing'] | p['EMSC_Smoothing']) / \
            (p['1st_derivative'] | p['2nd_derivative'] | p['RNV_Smoothing'])
    grid = grid + theme(figure_size=(24, 10))
    grid.save(f"./figures/PCA/year_pca_grid_{material}_category.svg", dpi=600)
