## load all the necessary libraries
import numpy as np
import pandas as pd

## load the data
data = pd.read_excel('./nirs/nirs_intact.xlsx')
### split the data
ref = data.iloc[:, np.r_[0:13]].set_index(data['Sample_ID'])
nirs_intact = data.iloc[:, 13:]

### create a dictionary for the nirs data
nirs = {
    "intact": nirs_intact.set_index(data['Sample_ID']),
    "ground": pd.read_excel('nirs/nirs_milled.xlsx').iloc[:, 1:].set_index(data['Sample_ID'])
}

### cut the spectra only to the range of 1100 - 2500 nm
for material in nirs.keys():
    nirs[material] = nirs[material].loc[:, (nirs[material].columns.astype(float) >= 1100) & (nirs[material].columns.astype(float) <= 2500)]

## create a dictionary for the y data
y = {
    "Year": ref['year'],
    "Country": ref['Country'],
    "DON": ref['Deoxynivalenol'],
    "Total_DON": ref['Total DON']
}

## Outlier Detection
from sklearn.decomposition import PCA  
from chemotools.outliers import HotellingT2, QResiduals 
outlier = {}
for material, matrix in nirs.items():
    matrix_array = matrix.to_numpy()
    pca = PCA(n_components=5).fit(matrix_array)
    hotelling_t2 = HotellingT2(model=pca, confidence=0.95).fit(matrix_array)
    q_residuals = QResiduals(model=pca, confidence=0.95).fit(matrix_array)
    outlier[material] ={
        'hotelling': hotelling_t2.predict(matrix_array),
        'q_residuals': q_residuals.predict(matrix_array)
    }
outlier_both = {}
for material in nirs.keys():
    df_outlier = pd.DataFrame(outlier[material])
    df_outlier.set_index(nirs[material].index, inplace=True)
    outlier_both[material] = df_outlier[(df_outlier['hotelling'] == -1) & (df_outlier['q_residuals'] == -1)]

## remove the outlier from the spectra and export the as cleaned data
nirs_cleaned = {
    "intact": nirs['intact'].drop(outlier_both['intact'].index),
    "ground": nirs['ground'].drop(outlier_both['ground'].index)
}


y_cleaned = {}
for material in outlier_both.keys():
    y_cleaned[material] = {}
    for toxin in y.keys():
        y_cleaned[material][toxin] = y[toxin].drop(outlier_both[material].index)


### create a long format dataframe for plotting
def format_spectra(df, data_name):
    #Reset index and melt into the long format
    long_df = df.reset_index().melt(
        id_vars='Sample_ID', 
        var_name='Wavelength', 
        value_name= "Reflectance (1/R)"
    )
    
    #Convert 'Sample_ID' to a categorical factor
    long_df['Sample_ID'] = long_df['Sample_ID'].astype('category')
    
    #Convert 'Wavelength' to a clean numeric format
    long_df['Wavelength'] = pd.to_numeric(long_df['Wavelength'], errors='coerce')
    
    return long_df

formatted_spectra = {}
for material in nirs_cleaned.keys():
    formatted_spectra[material] = format_spectra(nirs_cleaned[material], material)

formatted_spectra_before = {}
for material in nirs.keys():
    formatted_spectra_before[material] = format_spectra(nirs[material], material)

### create a plot function
def plot_outlier_spectra(spectra_before, spectra_after):
    def plot_spectra(spectra, title):
        from plotnine import ggplot, aes, geom_line, labs, theme_classic, theme, element_text, scale_color_cmap_d
        output = (
            ggplot(spectra, aes(x='Wavelength', y=spectra.columns[2], group='Sample_ID', color = "Sample_ID")) +
            geom_line(size = 0.5) +
            labs(x= "Wavelength (nm)", y=spectra.columns[2], title = title) +
            scale_color_cmap_d(cmap_name="viridis_r") +
            theme_classic() +
            theme(
                legend_position="none",
                title=element_text(size=14),
                axis_title=element_text(size=12)
            )
        )
        return output
    p1 = plot_spectra(spectra_before, " Before Outlier Removal ")
    p2 = plot_spectra(spectra_after, " After Outlier Removal ")
    export = p1/p2
    return export

p1 = plot_outlier_spectra(formatted_spectra_before['intact'], formatted_spectra['intact'])
p1.save(
    "figures/outlier_removal/outlier_spectra_intact.svg",
    width=20,
    height=8,
    units="in",
    dpi=300
)
p2 = plot_outlier_spectra(formatted_spectra_before['ground'], formatted_spectra['ground'])
p2.save(
    "figures/outlier_removal/outlier_spectra_ground.svg",
    width=20,
    height=8,
    units="in",
    dpi=300
)


## export all data as csv files
for material in nirs_cleaned.keys():
    nirs_cleaned[material].to_csv(f'./nirs/outlier_removal/nirs_{material}_cleaned.csv')

for material in y_cleaned.keys():
    pd.DataFrame(y_cleaned[material]).to_csv(f'./nirs/outlier_removal/y_{material}_cleaned.csv')

## split the data into training and testing sets into 10 different splits using random sampling
from sklearn.model_selection import train_test_split

splits = {}
for i in range(10):
    splits[i] = {}
    for material in nirs_cleaned.keys():
        
        # Grab the original X and y
        X_data = nirs_cleaned[material]
        y_data = pd.DataFrame(y_cleaned[material])
        
        # (Notice how it now returns 6 items instead of 4)
        X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
            X_data, 
            y_data,
            X_data.index, 
            test_size=0.2,
            stratify=y_data['Country'],  # Stratify by 'Country' to maintain class distribution
            random_state= i
        )
        
        # Put all results on the dictionary
        splits[i][material] = {
            "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test
        }

# Loop through the dictionary, unpacking the 4 dataframes for each material
for i, split in splits.items():
    for material, data in split.items():
        data["X_train"].to_csv(f'./nirs/train_test_set/X_train_{material}_split_{i}.csv')
        data["X_test"].to_csv(f'./nirs/train_test_set/X_test_{material}_split_{i}.csv')
        data["y_train"].to_csv(f'./nirs/train_test_set/y_train_{material}_split_{i}.csv')
        data["y_test"].to_csv(f'./nirs/train_test_set/y_test_{material}_split_{i}.csv')

