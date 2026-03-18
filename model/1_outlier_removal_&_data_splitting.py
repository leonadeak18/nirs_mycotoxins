## load all the necessary libraries
import numpy as np
import pandas as pd

## load the data
data = pd.read_excel('./nirs/SD_Foss_intact.xlsx')
### split the data
ref = data.iloc[:, np.r_[0, 3:13]].set_index(data['Sample_ID'])
nirs_intact = data.iloc[:, 13:]

### replace 0 to be 10 to avoid log(0) issue
ref = ref.replace(0, 10)

### create a dictionary for the nirs data
nirs = {
    "intact": nirs_intact.set_index(data['Sample_ID']),
    "ground": pd.read_excel('nirs/SD_Foss_milled.xlsx').iloc[:, 1:].set_index(data['Sample_ID'])
}
### add absorbance Value
nirs.update({
    "abs_intact": -np.log(nirs['intact']),
    "abs_ground": -np.log(nirs['ground'])
})

y = {
    "DON": ref['Deoxynivalenol'],
    "Total_DON": ref['Total DON'],
    "Log_DON": np.log(ref['Deoxynivalenol']),
    "Log_Total_DON": np.log(ref['Total DON'])
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
    "ground": nirs['ground'].drop(outlier_both['ground'].index),
    "abs_intact": nirs['abs_intact'].drop(outlier_both['abs_intact'].index),
    "abs_ground": nirs['abs_ground'].drop(outlier_both['abs_ground'].index)
}


y_cleaned = {}
for material in outlier_both.keys():
    y_cleaned[material] = {}
    for toxin in y.keys():
        y_cleaned[material][toxin] = y[toxin].drop(outlier_both[material].index)


## visualize before and after outlier removal
from plotnine import *
### create a long format dataframe for plotting
def format_spectra(df, data_name):
    """
    Takes a wide-format spectra DataFrame, melts it to long format, 
    and sets the correct data types for plotnine.
    """
    if data_name not in ['abs_intact', 'abs_ground']:
        unit = "Reflectance (1/R)"
    else:
        unit = "Absorbance (A)"

    #Reset index and melt into the long format
    long_df = df.reset_index().melt(
        id_vars='Sample_ID', 
        var_name='Wavelength', 
        value_name= unit
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
    "figures/outlier_spectra_intact.svg",
    width=20,
    height=10,
    units="in",
    dpi=300
)
p2 = plot_outlier_spectra(formatted_spectra_before['ground'], formatted_spectra['ground'])
p2.save(
    "figures/outlier_spectra_ground.svg",
    width=20,
    height=10,
    units="in",
    dpi=300
)

p3 = plot_outlier_spectra(formatted_spectra_before['abs_intact'], formatted_spectra['abs_intact'])
p3.save(
    "figures/outlier_spectra_abs_intact.svg",
    width=20,
    height=10,
    units="in",
    dpi=300
)

p4 = plot_outlier_spectra(formatted_spectra_before['abs_ground'], formatted_spectra['abs_ground'])
p4.save(
    "figures/outlier_spectra_abs_ground.svg",
    width=20,
    height=10,
    units="in",
    dpi=300
)


## export all data as csv files
for material in nirs_cleaned.keys():
    nirs_cleaned[material].to_csv(f'./nirs/outlier_removal/nirs_{material}_cleaned.csv')

for material in y_cleaned.keys():
    pd.DataFrame(y_cleaned[material]).to_csv(f'./nirs/outlier_removal/y_{material}_cleaned.csv')

## split the data into training and testing sets
from astartes import train_test_split
splits = {}

for material in nirs_cleaned.keys():
    
    # 1. Grab your original X and y data for clean reading
    X_data = nirs_cleaned[material]
    y_data = pd.DataFrame(y_cleaned[material])
    
    # 2. Run the split, passing the index as a THIRD array!
    # (Notice how it now returns 6 items instead of 4)
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X_data, 
        y_data, 
        X_data.index,
        test_size=0.2, 
        sampler="kennard_stone"
    )
    
    # 3. Rebuild the pandas DataFrames using those perfectly perfectly split indices
    splits[material] = (
        pd.DataFrame(X_train, index=idx_train, columns=X_data.columns),
        pd.DataFrame(X_test, index=idx_test, columns=X_data.columns),
        pd.DataFrame(y_train, index=idx_train, columns=y_data.columns),
        pd.DataFrame(y_test, index=idx_test, columns=y_data.columns)
    )

# Loop through the dictionary, unpacking the 4 dataframes for each material
for material, (X_train, X_test, y_train, y_test) in splits.items():
    X_train.to_csv(f'./nirs/train_test_set/X_train_{material}.csv')
    X_test.to_csv(f'./nirs/train_test_set/X_test_{material}.csv')
    y_train.to_csv(f'./nirs/train_test_set/y_train_{material}.csv')
    y_test.to_csv(f'./nirs/train_test_set/y_test_{material}.csv')

