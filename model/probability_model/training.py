import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import precision_recall_curve

# ==========================================
# 1. GENERATE DUMMY INFRARED-LIKE DATA
# ==========================================
print("Generating synthetic spectral data...")
X, y = make_classification(
    n_samples=1000, 
    n_features=300, 
    n_informative=30, 
    n_redundant=250, 
    random_state=42,
    weights=[0.5, 0.5]
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ==========================================
# 2. BUILD THE PIPELINE
# ==========================================
pca = PCA(n_components=15, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
calibrated_rf = CalibratedClassifierCV(rf, cv=3)

model_pipeline = Pipeline([
    ('pca', pca),
    ('clf', calibrated_rf)
])

# ==========================================
# 3. ROBUST THRESHOLD FINDING (100 REPEATS)
# ==========================================
n_repeats = 100
target_precision = 0.95

upper_thresholds = []
lower_thresholds = []

print(f"Running {n_repeats} CV iterations to find robust thresholds. This will take a moment...\n")

for i in range(n_repeats):
    # Fresh data shuffle for each iteration
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=i)
    
    # cross_val_predict automatically clones the pipeline, so no data leaks!
    cv_probs = cross_val_predict(model_pipeline, X_train, y_train, cv=cv_strategy, method='predict_proba')
    
    # Calculate Upper Threshold (High Contamination)
    prec_high, _, thresh_high = precision_recall_curve(y_train, cv_probs[:, 1])
    valid_idx_high = np.where(prec_high[:-1] >= target_precision)[0]
    if len(valid_idx_high) > 0:
        upper_thresholds.append(thresh_high[valid_idx_high[0]])
        
    # Calculate Lower Threshold (Low Contamination)
    prec_low, _, thresh_low = precision_recall_curve(1 - y_train, cv_probs[:, 0])
    valid_idx_low = np.where(prec_low[:-1] >= target_precision)[0]
    if len(valid_idx_low) > 0:
        lower_thresholds.append(1 - thresh_low[valid_idx_low[0]])

# Calculate the medians to ignore outliers
optimal_upper_thresh = np.median(upper_thresholds)
optimal_lower_thresh = np.median(lower_thresholds)

print("-" * 45)
print(f"Target Precision: {target_precision * 100}%")
print(f"Robust Lower Threshold (Low Contam)  : < {optimal_lower_thresh:.3f}")
print(f"Robust Upper Threshold (High Contam) : > {optimal_upper_thresh:.3f}")
print("-" * 45)

# ==========================================
# 4. VISUALIZE THE THRESHOLD DISTRIBUTIONS
# ==========================================
plt.figure(figsize=(10, 6))

# Plot histograms of the 100 collected thresholds
plt.hist(lower_thresholds, bins=15, alpha=0.6, color='blue', label='Lower Thresholds (Low Contam)')
plt.hist(upper_thresholds, bins=15, alpha=0.6, color='red', label='Upper Thresholds (High Contam)')

# Add vertical lines for the final chosen median thresholds
plt.axvline(optimal_lower_thresh, color='darkblue', linestyle='dashed', linewidth=2, 
            label=f'Median Lower: {optimal_lower_thresh:.3f}')
plt.axvline(optimal_upper_thresh, color='darkred', linestyle='dashed', linewidth=2, 
            label=f'Median Upper: {optimal_upper_thresh:.3f}')

# Shade the "Uncertain" area
plt.axvspan(optimal_lower_thresh, optimal_upper_thresh, color='gray', alpha=0.2, label='Uncertain Zone')

plt.title(f'Distribution of 95% Precision Thresholds over {n_repeats} CV Repeats')
plt.xlabel('Probability Score')
plt.ylabel('Frequency')
plt.xlim(0, 1)
plt.legend(loc='upper center')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# ==========================================
# 5. TRAIN FINAL MODEL & APPLY THRESHOLDS
# ==========================================
print("\nTraining final model on all training data...")
model_pipeline.fit(X_train, y_train)

def classify_with_uncertainty(model, X_data, lower_t, upper_t):
    probs = model.predict_proba(X_data)[:, 1]
    predictions = np.full(probs.shape, 'Uncertain', dtype=object)
    predictions[probs < lower_t] = 'Low Contamination'
    predictions[probs > upper_t] = 'High Contamination'
    return predictions, probs

custom_labels, test_probs = classify_with_uncertainty(
    model_pipeline, X_test, optimal_lower_thresh, optimal_upper_thresh
)

results_df = pd.DataFrame({
    'True_Label': ['High Contamination' if val == 1 else 'Low Contamination' for val in y_test],
    'Model_Prediction': custom_labels,
    'Probability_Score': test_probs
})

print("\n--- Summary of Test Set Predictions ---")
print(results_df['Model_Prediction'].value_counts())