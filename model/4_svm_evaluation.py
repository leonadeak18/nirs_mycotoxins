##################################
###### SVM evaluation #########
##################################
import numpy as np
from scipy.stats import mannwhitneyu, ttest_ind, shapiro
import pandas as pd
from plotnine import (ggplot, aes, geom_boxplot, geom_jitter, geom_text, 
                      theme_classic, labs, theme, element_text, expand_limits, scale_y_continuous)

#------- Determine the best model by boxplot -------#
# Importing the datasets
data = {
    "weighted": pd.read_csv("./results/mixed_all/weighted_SVM_results.csv"),
    "non_weighted": pd.read_csv("./results/mixed_all/2_SVM_results.csv")
}

# rename the columns "Test_MCC" to "MCC" for consistency
for key in data.keys():
    data[key] = data[key].rename(columns={"Test_MCC": "MCC"})

data_top = {}
for key in data.keys():
    # Create the model label
    data[key]['model'] = data[key]['Target_Toxin'] + " (" + data[key]['Sample_Source'] + ")\n" + data[key]['Preprocessing'] + " \n " + data[key]["Feature_Method"]
    
    # Group by model to calculate metrics
    grouped_mcc = data[key].groupby('model')['MCC']
    
    # Calculate the score: Mean - Standard Deviation
    # fillna(0) ensures that if a model only has 1 sample (std = NaN), it doesn't break the sort
    mcc_score = grouped_mcc.mean() - grouped_mcc.std().fillna(0)
    
    # Determine the Rank order (Best to Worst based on Mean - SD)
    rank_order = mcc_score.sort_values(ascending=False).index.tolist()
    
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
        + scale_y_continuous(limits=(0.3, 0.9))
        + labs(
            title=f"Top Models Comparison ({key.replace('_', ' ').title()})", 
            x="Model Configuration (Rank 1 on the Left)", 
            y="Matthews Correlation Coefficient (MCC)"
        )
        + theme(
            axis_text_x=element_text(size=9),
            plot_title=element_text(size=14, weight='bold')
        )
    )

# export the plots
for key, plot in plots.items():
    plot.save(f"./figures/SVM/{key}_svm_top_models_boxplot.svg", width=10, height=6, dpi=300)


#------- Comparison the best model -------#
## the best model is the weighted Total_DON in intact with SNV_Smoothing preprocessing, we will compare it with other effects
best_model_str = "Total_DON (intact)\nSNV_Smoothing \n ICA"

# Function to calculate statistical significance with dynamic test selection
def get_sig_df(df, ref_name, group_col='model', val_col='MCC'):
    ref_vals = df[df[group_col] == ref_name].sort_values('Split')[val_col].values
    groups = df[group_col].unique()
    
    stats_list = []
    for g in groups:
        if g == ref_name:
            stats_list.append({group_col: g, 'p_label': '', 'y_pos': df[df[group_col] == g][val_col].max() + 0.05})
            continue
            
        comp_vals = df[df[group_col] == g].sort_values('Split')[val_col].values
        
        try:
            # 1. Normality Test (Shapiro-Wilk)
            # Requires at least 3 samples to run the test
            if len(ref_vals) >= 3 and len(comp_vals) >= 3:
                _, p_norm_ref = shapiro(ref_vals)
                _, p_norm_comp = shapiro(comp_vals)
                
                # If p >= 0.05, we fail to reject the null hypothesis (data is normally distributed)
                is_normal = (p_norm_ref >= 0.05) and (p_norm_comp >= 0.05)
            else:
                # If we don't have enough data to prove normality, default to non-parametric
                is_normal = False
            
            # 2. Select and run the appropriate statistical test
            if is_normal:
                # Use Welch's t-test (equal_var=False) as a safer default for normally distributed data
                _, p = ttest_ind(ref_vals, comp_vals, equal_var=False)
                print(f"Comparing {ref_name} vs {g} using Welch's t-test: p-value = {p:.4f}")
            else:
                # Fallback to Mann-Whitney U for non-normal distributions
                _, p = mannwhitneyu(ref_vals, comp_vals, alternative='two-sided')
                print(f"Comparing {ref_name} vs {g} using Mann-Whitney U test: p-value = {p:.4f}")
            
            # 3. Assign significance stars
            if p < 0.001: star = "***"
            elif p < 0.01: star = "**"
            elif p < 0.05: star = "*"
            else: star = "ns"
            
        except ValueError:
            star = "n/a"
        
        stats_list.append({group_col: g, 'p_label': star, 'y_pos': df[df[group_col] == g][val_col].max() + 0.05})
        
    return pd.DataFrame(stats_list)

#------- Define the 6 Comparisons -------#

# Plot 1: Toxin Effect (Total_DON vs DON)
df1 = data["weighted"][data["weighted"]['model'].isin([best_model_str, "DON (intact)\nSNV_Smoothing \n ICA"])].copy()
df1['model'] = pd.Categorical(df1['model'], categories=[best_model_str, "DON (intact)\nSNV_Smoothing \n ICA"])

# Plot 2: Sample Source Effect (Intact vs Ground)
df2 = data["weighted"][data["weighted"]['model'].isin([best_model_str, "Total_DON (ground)\nSNV_Smoothing \n ICA"])].copy()
df2['model'] = pd.Categorical(df2['model'], categories=[best_model_str, "Total_DON (ground)\nSNV_Smoothing \n ICA"])

# Plot 3: Preprocessing Effect (SNV vs the others)
prep_models = [best_model_str, 
               "Total_DON (intact)\nRNV_Smoothing \n ICA", 
               "Total_DON (intact)\n1st_derivative \n ICA", 
               "Total_DON (intact)\n2nd_derivative \n ICA", 
               "Total_DON (intact)\nDetrending_SNV \n ICA"]
df3 = data["weighted"][data["weighted"]['model'].isin(prep_models)].copy()
df3['model'] = pd.Categorical(df3['model'], categories=prep_models)


# Plot 4: Weighting Effect
df4_w = data["weighted"][data["weighted"]['model'] == best_model_str].copy()
df4_w['model'] = f"{best_model_str} Weighted"
df4_nw = data["non_weighted"][data["non_weighted"]['model'] == best_model_str].copy()
df4_nw['model'] = f"{best_model_str} Non-Weighted"
df4 = pd.concat([df4_w, df4_nw])
df4['model'] = pd.Categorical(df4['model'], categories=[f"{best_model_str} Weighted", f"{best_model_str} Non-Weighted"])

# Plot 5: Weighting Effect (the best vs the best models)
df5_w = data["weighted"][data["weighted"]['model'] == best_model_str].copy()
df5_w['model'] = f"{best_model_str} Weighted"
df5_nw = data["non_weighted"][data["non_weighted"]['model'] == "Total_DON (intact)\nSNV_Smoothing \n Factor_Analysis"].copy()
df5_nw['model'] = "Total_DON (intact)\nSNV_Smoothing \n Factor_Analysis Non-Weighted"
df5 = pd.concat([df5_w, df5_nw])
df5['model'] = pd.Categorical(df5['model'], categories=[f"{best_model_str} Weighted", "Total_DON (intact)\nSNV_Smoothing \n Factor_Analysis Non-Weighted"])

#Plot 6: Effect of Feature Selection (Factor Analysis vs PCA)
fs_models = [best_model_str, 
               "Total_DON (intact)\nSNV_Smoothing \n Full_spectra", 
               "Total_DON (intact)\nSNV_Smoothing \n ANOVA_Percentile", 
               "Total_DON (intact)\nSNV_Smoothing \n ANOVA_FDR", 
               "Total_DON (intact)\nSNV_Smoothing \n Factor_Analysis",
               "Total_DON (intact)\nSNV_Smoothing \n PCA",
               "Total_DON (intact)\nSNV_Smoothing \n Lasso L1 (Selector)",
               "Total_DON (intact)\nSNV_Smoothing \n RFE"]
df6 = data["weighted"][data["weighted"]['model'].isin(fs_models)].copy()
df6['model'] = pd.Categorical(df6['model'], categories=fs_models)

#------- Generate and Print Plots -------#

comp_list = [
    (df1, "Toxin Effect", "Best Model\n(Weighted)"),
    (df2, "Material Effect", "Best Model\n(Weighted)"),
    (df3, "Preprocessing Effect", "Best Model\n(Weighted)"),
    (df4, "Weighting Effect", "Best Model\n(Weighted)"),
    (df5, "Weighting Effect (Best vs Best)", "Best Model\n(Weighted)"),
    (df6, "Feature Selection Effect", "Best Model\n(Weighted)")
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
         + theme(axis_text_x=element_text(), figure_size=(9, 5))
    )

#export the comparison plots
for key, plot in p_comp.items():
    plot.save(f"./figures/SVM/{key.replace(' ', '_').lower()}_comparison_boxplot.svg", width=9, height=5, dpi=300)
