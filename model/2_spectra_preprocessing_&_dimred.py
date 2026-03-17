<<<<<<< HEAD
## Import all libraries
import numpy as np
import pandas as pd


## load the data
test_set = {
    "intact" : pd.read_csv("nirs/outlier_removal/X_test_intact.csv", index_col=0),
    "ground" : pd.read_csv("nirs/outlier_removal/X_test_ground.csv", index_col=0)
}
train_set = {
    "intact" : pd.read_csv("nirs/outlier_removal/X_train_intact.csv", index_col=0),
    "ground" : pd.read_csv("nirs/outlier_removal/X_train_ground.csv", index_col=0)
}

## Preprocessing
### custom transformers for spectra preprocessing
import pywt
from sklearn.base import BaseEstimator, TransformerMixin
class WaveletDenoising(BaseEstimator, TransformerMixin):
    def __init__(self, wavelet='db1', level=2):
        self.wavelet = wavelet
        self.level = level

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X, y=None):
        X_transformed = np.empty_like(X)
        for i in range(X.shape[0]):
            spectrum = X[i, :]
            
            # Decompose the signal
            coeffs = pywt.wavedec(spectrum, self.wavelet, level=self.level)
            
            # Threshold the detail coefficients to remove noise
            sigma = np.median(np.abs(coeffs[-1])) / 0.6745
            threshold = sigma * np.sqrt(2 * np.log(len(spectrum)))
            coeffs[1:] = [pywt.threshold(c, value=threshold, mode='soft') for c in coeffs[1:]]
            
            # Reconstruct the signal back to the original length
            reconstructed = pywt.waverec(coeffs, self.wavelet)
            if len(reconstructed) > X.shape[1]:
                reconstructed = reconstructed[:X.shape[1]]
                
            X_transformed[i, :] = reconstructed
        return X_transformed

### Import all preprocessing methods
from chemotools.smooth import SavitzkyGolayFilter
from chemotools.scatter import StandardNormalVariate, MultiplicativeScatterCorrection, ExtendedMultiplicativeScatterCorrection, RobustNormalVariate
from chemotools.derivative import SavitzkyGolay
from sklearn.pipeline import make_pipeline

def spectra_preprocessing(train_dict, test_dict):
    # Initialized the results dictionaries
    train_results = {}
    test_results = {}
    
    # Precalculate for reference from training data
    reference_intact = train_dict["intact"].to_numpy().mean(axis=0)
    reference_ground = train_dict["ground"].to_numpy().mean(axis=0)
    
    # Loop through each material ("intact", "ground")
    for material in train_dict.keys():
        train_df = train_dict[material]
        test_df = test_dict[material]
        
        # Select the correct reference baseline for this specific material
        current_reference = reference_intact if material == "intact" else reference_ground
        
        # Define the transformers dynamically for this material
        transformers = {
            "Smoothing": SavitzkyGolayFilter(window_size=15, polynomial_order=3),
            
            "SNV": make_pipeline(
                SavitzkyGolayFilter(window_size=15, polynomial_order=3), 
                StandardNormalVariate()
            ),
            
            "MSC": make_pipeline(
                SavitzkyGolayFilter(window_size=15, polynomial_order=3),
                MultiplicativeScatterCorrection(reference=current_reference)
            ),
            
            "EMSC": make_pipeline(
                SavitzkyGolayFilter(window_size=15, polynomial_order=3),
                ExtendedMultiplicativeScatterCorrection(reference=current_reference)
            ),
            
            "1st_Deriv": SavitzkyGolay(window_size=15, polynomial_order=3, derivate_order=1),
            
            "2nd_Deriv": SavitzkyGolay(window_size=15, polynomial_order=3, derivate_order=2),

            "RNV": make_pipeline(
                SavitzkyGolayFilter(window_size=15, polynomial_order=3), 
                RobustNormalVariate()
            ),

            "0_wavelets": make_pipeline(
                SavitzkyGolayFilter(window_size=15, polynomial_order=3),
                WaveletDenoising(wavelet='db1', level=2)
            ),

            "1st_Deriv_wavelets": make_pipeline(
                SavitzkyGolay(window_size=15, polynomial_order=3, derivate_order=1),
                WaveletDenoising(wavelet='db1', level=2)
            ),
            "2nd_Deriv_wavelets": make_pipeline(
                SavitzkyGolay(window_size=15, polynomial_order=3, derivate_order=2),
                WaveletDenoising(wavelet='db1', level=2)
            )
        }
        
        # 5. Apply every transformation safely
        for prep_name, transformer in transformers.items():
            
            # Create the dictionary keys if they don't exist yet
            if prep_name not in train_results:
                train_results[prep_name] = {}
                test_results[prep_name] = {}
            
            # Fit & Transform the training data
            train_array = transformer.fit_transform(train_df.to_numpy())
            train_results[prep_name][material] = pd.DataFrame(
                train_array, index=train_df.index, columns=train_df.columns
            )
            
            # Transform ONLY on the test data
            test_array = transformer.transform(test_df.to_numpy())
            test_results[prep_name][material] = pd.DataFrame(
                test_array, index=test_df.index, columns=test_df.columns
            )
            
    return train_results, test_results

# Execute the function!
train_preprocessed, test_preprocessed = spectra_preprocessing(train_set, test_set)


## Dimensionality reduction
=======
## Import all libraries
import numpy as np
import pandas as pd


## load the data
test_set = {
    "intact" : pd.read_csv("nirs/outlier_removal/X_test_intact.csv", index_col=0),
    "ground" : pd.read_csv("nirs/outlier_removal/X_test_ground.csv", index_col=0)
}
train_set = {
    "intact" : pd.read_csv("nirs/outlier_removal/X_train_intact.csv", index_col=0),
    "ground" : pd.read_csv("nirs/outlier_removal/X_train_ground.csv", index_col=0)
}

## Preprocessing
### custom transformers for spectra preprocessing
import pywt
from sklearn.base import BaseEstimator, TransformerMixin
class WaveletDenoising(BaseEstimator, TransformerMixin):
    def __init__(self, wavelet='db1', level=2):
        self.wavelet = wavelet
        self.level = level

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X, y=None):
        X_transformed = np.empty_like(X)
        for i in range(X.shape[0]):
            spectrum = X[i, :]
            
            # Decompose the signal
            coeffs = pywt.wavedec(spectrum, self.wavelet, level=self.level)
            
            # Threshold the detail coefficients to remove noise
            sigma = np.median(np.abs(coeffs[-1])) / 0.6745
            threshold = sigma * np.sqrt(2 * np.log(len(spectrum)))
            coeffs[1:] = [pywt.threshold(c, value=threshold, mode='soft') for c in coeffs[1:]]
            
            # Reconstruct the signal back to the original length
            reconstructed = pywt.waverec(coeffs, self.wavelet)
            if len(reconstructed) > X.shape[1]:
                reconstructed = reconstructed[:X.shape[1]]
                
            X_transformed[i, :] = reconstructed
        return X_transformed

### Import all preprocessing methods
from chemotools.smooth import SavitzkyGolayFilter
from chemotools.scatter import StandardNormalVariate, MultiplicativeScatterCorrection, ExtendedMultiplicativeScatterCorrection, RobustNormalVariate
from chemotools.derivative import SavitzkyGolay
from sklearn.pipeline import make_pipeline

def spectra_preprocessing(train_dict, test_dict):
    # Initialized the results dictionaries
    train_results = {}
    test_results = {}
    
    # Precalculate for reference from training data
    reference_intact = train_dict["intact"].to_numpy().mean(axis=0)
    reference_ground = train_dict["ground"].to_numpy().mean(axis=0)
    
    # Loop through each material ("intact", "ground")
    for material in train_dict.keys():
        train_df = train_dict[material]
        test_df = test_dict[material]
        
        # Select the correct reference baseline for this specific material
        current_reference = reference_intact if material == "intact" else reference_ground
        
        # Define the transformers dynamically for this material
        transformers = {
            "Smoothing": SavitzkyGolayFilter(window_size=15, polynomial_order=3),
            
            "SNV": make_pipeline(
                SavitzkyGolayFilter(window_size=15, polynomial_order=3), 
                StandardNormalVariate()
            ),
            
            "MSC": make_pipeline(
                SavitzkyGolayFilter(window_size=15, polynomial_order=3),
                MultiplicativeScatterCorrection(reference=current_reference)
            ),
            
            "EMSC": make_pipeline(
                SavitzkyGolayFilter(window_size=15, polynomial_order=3),
                ExtendedMultiplicativeScatterCorrection(reference=current_reference)
            ),
            
            "1st_Deriv": SavitzkyGolay(window_size=15, polynomial_order=3, derivate_order=1),
            
            "2nd_Deriv": SavitzkyGolay(window_size=15, polynomial_order=3, derivate_order=2),

            "RNV": make_pipeline(
                SavitzkyGolayFilter(window_size=15, polynomial_order=3), 
                RobustNormalVariate()
            ),

            "0_wavelets": make_pipeline(
                SavitzkyGolayFilter(window_size=15, polynomial_order=3),
                WaveletDenoising(wavelet='db1', level=2)
            ),

            "1st_Deriv_wavelets": make_pipeline(
                SavitzkyGolay(window_size=15, polynomial_order=3, derivate_order=1),
                WaveletDenoising(wavelet='db1', level=2)
            ),
            "2nd_Deriv_wavelets": make_pipeline(
                SavitzkyGolay(window_size=15, polynomial_order=3, derivate_order=2),
                WaveletDenoising(wavelet='db1', level=2)
            )
        }
        
        # 5. Apply every transformation safely
        for prep_name, transformer in transformers.items():
            
            # Create the dictionary keys if they don't exist yet
            if prep_name not in train_results:
                train_results[prep_name] = {}
                test_results[prep_name] = {}
            
            # Fit & Transform the training data
            train_array = transformer.fit_transform(train_df.to_numpy())
            train_results[prep_name][material] = pd.DataFrame(
                train_array, index=train_df.index, columns=train_df.columns
            )
            
            # Transform ONLY on the test data
            test_array = transformer.transform(test_df.to_numpy())
            test_results[prep_name][material] = pd.DataFrame(
                test_array, index=test_df.index, columns=test_df.columns
            )
            
    return train_results, test_results

# Execute the function!
train_preprocessed, test_preprocessed = spectra_preprocessing(train_set, test_set)


## Dimensionality reduction
>>>>>>> c0b013e83447d4824ced8d39df8883c0ce891429
from sklearn.decomposition import PCA, NMF, FastICA