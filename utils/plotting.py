## import alll necessary libraries
from plotly.graph_objs.layout import xaxis
import pandas as pd
import plotly.express as px


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


import plotly.express as px

def plot_spectra(df):
    fig = px.line(
        df,
        x='Wavelength',
        y='Reflectance (1/R)',
        color='Sample_ID',
        hover_data=['Sample_ID'], 
        color_discrete_sequence=px.colors.sequential.Viridis, 
        labels={
            'Wavelength': 'Wavelength (nm)', 
            'Reflectance (1/R)': 'Reflectance (1/R)'
        }
    )
    
    # Apply layout improvements for a crisp, publication-ready look
    fig.update_layout(
        template='simple_white', 
        showlegend = False,
        xaxis=dict(dtick = 200),
        font=dict(family="Arial", size=12) 
    )
    
    # Ensure clean lines and a helpful hover tooltip
    fig.update_traces(
        mode='lines', 
        line=dict(width=1.5), 
        hovertemplate=(
            "<b>Sample:</b> %{customdata[0]}<br>" + 
            "<b>Wavelength:</b> %{x} nm<br>" +
            "<b>Reflectance:</b> %{y:.4f}<extra></extra>"
        )
    )

    return fig
