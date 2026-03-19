## import alll necessary libraries
from plotnine import scale_x_continuous
import pandas as pd
from plotnine import *


def format_spectra(df, data_name):
    """
    Takes a wide-format spectra DataFrame, melts it to long format, 
    and sets the correct data types for plotnine.
    """
    if data_name not in ['abs_intact', 'abs_ground']:
        unit = "Reflectance (1/R)"
    else:
        unit = "Absorbance (A)"

    #Ensure the index is named 'Sample_ID' for melting
    if df.index.name is None:
        df = df.copy()
        df.index.name = 'Sample_ID'

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

## spectra plotting function
def plot_spectra(spectra, title, color_var = "Sample_ID"):
    num_categories = spectra[color_var].nunique()
    base_plot = (
        ggplot(spectra, aes(x='Wavelength', y=spectra.columns[2], group='Sample_ID', color = color_var)) +
        geom_line(size = 0.5) +
        labs(x= "Wavelength (nm)", y=spectra.columns[2], title = title) +
        scale_x_continuous(breaks=range(400, 2500, 300)) +
        theme_classic() +
        theme(
            title=element_text(size=12),
            axis_title=element_text(size=10)
            )
        )
    if num_categories == 2:
        output = base_plot + scale_color_manual(values=['#70c8e2', '#e54b2c']) + theme(legend_position = "right")
    elif num_categories == 3:
        output = base_plot + scale_color_manual(values=['#70c8e2', 'black', '#e54b2c']) + theme(legend_position = "right")
    else:
        output = base_plot + scale_color_cmap_d(cmap_name="viridis_r") + theme(legend_position = "none")

    return output

## PCA plotting function
def plot_pca(pca_df, variance_explained, title, color_var):
    num_categories = pca_df[color_var].nunique()
    base_plot = (
        ggplot(pca_df, aes(x='PC1', y='PC2', color=color_var)) +
        geom_point(size=3) +
        labs(x=f"PC1 ({variance_explained[0]*100:.2f}%)", y=f"PC2 ({variance_explained[1]*100:.2f}%)", title=title) +
        theme_classic() +
        theme(
            title=element_text(size=12),
            axis_title=element_text(size=10)
            )
        )
    if num_categories == 2:
        output = base_plot + scale_color_manual(values=['#70c8e2', '#e54b2c']) + theme(legend_position = "right")
    elif num_categories == 3:
        output = base_plot + scale_color_manual(values=['#70c8e2', 'black', '#e54b2c']) + theme(legend_position = "right")
    else:
        output = base_plot + scale_color_cmap_d(cmap_name="viridis_r") + theme(legend_position = "none")

    return output
