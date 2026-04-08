##################################
###### PLS-DA evaluation #########
##################################
import numpy as np
from scipy.stats import wilcoxon
import pandas as pd
from plotnine import (ggplot, aes, geom_boxplot, geom_jitter, geom_text, 
                      theme_classic, labs, theme, element_text, expand_limits)

#------- Determine the best model by boxplot -------#
# Importing the datasets
data = {
    "weighted": pd.read_csv("./results/mixed_all/weighted_plsda_results.csv"),
    "non_weighted": pd.read_csv("./results/mixed_all/2_plsda_results.csv")
}

data_top = {}
for key in data.keys():
    # Create the model label
    data[key]['model'] = data[key]['Target_toxin'] + " (" + data[key]['Sample_source'] + ")\n" + data[key]['Preprocessing']
    
    # Determine the Rank order (Best to Worst based on Mean MCC)
    rank_order = data[key].groupby('model')['MCC'].mean().sort_values(ascending=False).index.tolist()
    
    # Filter to get the top 5 models
    top_models = rank_order[:5]
    df_filtered = data[key][data[key]['model'].isin(top_models)].copy()
    
    # Set the categories in the specific rank order
    df_filtered['model'] = pd.Categorical(df_filtered['model'], categories=top_models, ordered=True)
    
    data_top[key] = df_filtered

# Build the boxplots
plots = {}
for key in data_top.keys():
    plots[key] = (
        ggplot(data_top[key], aes(x='model', y='MCC', fill='model'))
        + geom_boxplot(show_legend=False, alpha=0.7)
        + geom_jitter(alpha=0.3, width=0.1, size=2, show_legend=False)
        + theme_classic()
        + labs(
            title=f"Top Models Comparison ({key.replace('_', ' ').title()})", 
            x="Model Configuration (Rank 1 on the Left)", 
            y="Matthews Correlation Coefficient (MCC)"
        )
        + theme(
            axis_text_x=element_text( size=9),
            plot_title=element_text(size=14, weight='bold')
        )
    )

# export the plots
for key, plot in plots.items():
    plot.save(f"./figures/PLS_DA/{key}_plsda_top_models_boxplot.svg", width=10, height=6, dpi=300)


#------- Comparison the best model -------#
## the best model is the weighted Total_DON in intact with SNV preprocessing, we will compare it with other effects
best_model_str = "Total_DON (intact)\nSNV"

# Function to calculate Wilcoxon stars compared to a reference
def get_sig_df(df, ref_name, group_col='model', val_col='MCC'):
    ref_vals = df[df[group_col] == ref_name].sort_values('Split')[val_col].values
    groups = df[group_col].unique()
    
    stats_list = []
    for g in groups:
        if g == ref_name:
            stats_list.append({group_col: g, 'p_label': '', 'y_pos': df[df[group_col] == g][val_col].max() + 0.05})
            continue
            
        comp_vals = df[df[group_col] == g].sort_values('Split')[val_col].values
        if len(ref_vals) == len(comp_vals):
            _, p = wilcoxon(ref_vals, comp_vals)
            if p < 0.001: star = "***"
            elif p < 0.01: star = "**"
            elif p < 0.05: star = "*"
            else: star = "ns"
        else:
            star = "n/a"
        
        stats_list.append({group_col: g, 'p_label': star, 'y_pos': df[df[group_col] == g][val_col].max() + 0.05})
    return pd.DataFrame(stats_list)

#------- Define the 4 Comparisons -------#

# Plot 1: Toxin Effect (Total_DON vs DON)
df1 = data["weighted"][data["weighted"]['model'].isin([best_model_str, "DON (intact)\nSNV"])].copy()
df1['model'] = pd.Categorical(df1['model'], categories=[best_model_str, "DON (intact)\nSNV"])

# Plot 2: Sample Source Effect (Intact vs Ground)
df2 = data["weighted"][data["weighted"]['model'].isin([best_model_str, "Total_DON (ground)\nSNV"])].copy()
df2['model'] = pd.Categorical(df2['model'], categories=[best_model_str, "Total_DON (ground)\nSNV"])

# Plot 3: Preprocessing Effect (SNV vs the others)
prep_models = [best_model_str, "Total_DON (intact)\nDetrend_SNV", "Total_DON (intact)\n1st_Der", 
               "Total_DON (intact)\n2nd_Der", "Total_DON (intact)\nRNV"]
df3 = data["weighted"][data["weighted"]['model'].isin(prep_models)].copy()
df3['model'] = pd.Categorical(df3['model'], categories=prep_models)

# Plot 4: Weighting Effect
df4_w = data["weighted"][data["weighted"]['model'] == best_model_str].copy()
df4_w['model'] = f"{best_model_str} Weighted"
df4_nw = data["non_weighted"][data["non_weighted"]['model'] == best_model_str].copy()
df4_nw['model'] = f"{best_model_str} Non-Weighted"
df4 = pd.concat([df4_w, df4_nw])
df4['model'] = pd.Categorical(df4['model'], categories=[f"{best_model_str} Weighted", f"{best_model_str} Non-Weighted"])
#------- Generate and Print Plots -------#

comp_list = [
    (df1, "Toxin Effect", "Best Model\n(Weighted)"),
    (df2, "Material Effect", "Best Model\n(Weighted)"),
    (df3, "Preprocessing Effect", "Best Model\n(Weighted)"),
    (df4, "Weighting Effect", "Best Model\n(Weighted)")
]

p_comp = {}
for df_plot, title, ref in comp_list:
    sig_df = get_sig_df(df_plot, df_plot['model'].cat.categories[0])
    
    p_comp[title] = (ggplot(df_plot, aes(x='model', y='MCC', fill='model'))
         + geom_boxplot(show_legend=False, alpha=0.7)
         + geom_jitter(alpha=0.3, width=0.1, show_legend=False)
         + geom_text(data=sig_df, mapping=aes(label='p_label', y='y_pos'), size=12, va='bottom')
         + theme_classic()
         + expand_limits(y=df_plot['MCC'].max() + 0.15)
         + labs(title=title, x="", y="Matthews Correlation Coefficient")
         + theme(axis_text_x=element_text(), figure_size=(8, 5))
    )

# Export the comparison plots
for title, plot in p_comp.items():
    plot.save(f"./figures/PLS_DA/{title.replace(' ', '_').lower()}_comparison_boxplot.svg", width=8, height=5, dpi=300)
