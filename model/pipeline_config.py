# ==========================================
# pipeline_config.py
# ==========================================
import pandas as pd
from utils.preparation import binary_transform_with_weight, binary_transform
import chemotools.scatter as ch
from chemotools.derivative import SavitzkyGolay
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA, FastICA, FactorAnalysis
from sklearn.feature_selection import SelectPercentile, SelectFdr, f_classif, SelectFromModel, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (roc_auc_score, accuracy_score, recall_score, make_scorer)

# --- STATIC CONFIGURATIONS ---
preprocessing_pipelines = {
    "SNV_Smoothing": Pipeline([('snv', ch.StandardNormalVariate()), ('smoothing', SavitzkyGolay(window_length=15, polyorder=2, deriv=0))]),
    "EMSC_Smoothing": Pipeline([('emsc', ch.ExtendedMultiplicativeScatterCorrection()), ('smoothing', SavitzkyGolay(window_length=15, polyorder=2, deriv=0))]),
    "1st_derivative": Pipeline([('first_derivative', SavitzkyGolay(window_length=15, polyorder=2, deriv=1))]),
    "2nd_derivative": Pipeline([('second_derivative', SavitzkyGolay(window_length=15, polyorder=2, deriv=2))]),
    "RNV_Smoothing": Pipeline([('rnv', ch.RobustNormalVariate()), ('smoothing', SavitzkyGolay(window_length=15, polyorder=2, deriv=0))])
}

feature_pipelines = {
    "Full_spectra": None,
    "PCA": Pipeline([('PCA', PCA(n_components=0.99, random_state=270725))]),
    "ICA": Pipeline([('ICA', FastICA(n_components=30, max_iter=1000, random_state=270725))]),
    "Factor_Analysis": Pipeline([('Factor_Analysis', FactorAnalysis(n_components=30, random_state=270725))]),
    "ANOVA_Pencentile": Pipeline([('ANOVA_Percentile', SelectPercentile(score_func=f_classif, percentile=20))]),
    "ANOVA_FDR": Pipeline([('ANOVA_FDR', SelectFdr(score_func=f_classif, alpha=0.05))]),
    "Lasso L1 (Selector)": Pipeline([('scaler', StandardScaler()), ('Lasso_L1', SelectFromModel(LogisticRegression(penalty='l1', solver='liblinear', C=10, random_state=270725)))]),   
    "RFE": Pipeline([('scaler', StandardScaler()), ('RFE', RFE(estimator=SVC(kernel='linear', random_state=270726), n_features_to_select=50, step=0.1))])
}

scoring_metrics = {
    'roc_auc': make_scorer(roc_auc_score),
    'accuracy': make_scorer(accuracy_score),
    'sensitivity': make_scorer(recall_score), 
    'specificity': make_scorer(recall_score, pos_label=0)
}

# --- DATA LOADING FUNCTION ---
def load_nirs_data(splits=10, threshold=1000, weighted = False):
    """Loads all CSVs, applies binary transforms, and extracts countries in one clean loop."""
    sources = ['intact', 'ground']
    targets = ['DON', 'Total_DON']
    
    X_train, X_test = {s: {} for s in sources}, {s: {} for s in sources}
    y_tr_bin, y_te_bin = {s: {} for s in sources}, {s: {} for s in sources}
    train_country, test_country = {s: {} for s in sources}, {s: {} for s in sources}

    for source in sources:
        for i in range(splits):
            # Load raw data
            X_train[source][i] = pd.read_csv(f'nirs/train_test_set/X_train_{source}_split_{i}.csv', index_col=0)
            X_test[source][i] = pd.read_csv(f'nirs/train_test_set/X_test_{source}_split_{i}.csv', index_col=0)
            
            y_train_raw = pd.read_csv(f'nirs/train_test_set/y_train_{source}_split_{i}.csv', index_col=0)
            y_test_raw = pd.read_csv(f'nirs/train_test_set/y_test_{source}_split_{i}.csv', index_col=0)
            
            # Transform to binary
            if weighted:
                y_tr_bin[source][i] = binary_transform_with_weight(y_train_raw, threshold=threshold, targets=targets)
                y_te_bin[source][i] = binary_transform_with_weight(y_test_raw, threshold=threshold, targets=targets)
            else:
                y_tr_bin[source][i] = binary_transform(y_train_raw, threshold=threshold, targets=targets)
                y_te_bin[source][i] = binary_transform(y_test_raw, threshold=threshold, targets=targets)
            
            # Extract country
            train_country[source][i] = y_train_raw['Country']
            test_country[source][i] = y_test_raw['Country']
            
    return X_train, X_test, y_tr_bin, y_te_bin, train_country, test_country