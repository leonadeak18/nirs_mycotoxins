import pandas as pd
import numpy as np

## The utility to transform continueous DON values into three classes: Low Contamination (LC), Uncertain (UC), and High Contamination (HC).
def three_level_transform(y, targets=['DON', 'Total_DON']):
    bins = [-float('inf'), 500, 1500, float('inf')]
    labels = ['LC', 'UC', 'HC']
    
    y_class = pd.DataFrame(index=y.index)
    
    for target in targets:
        y_class[target] = pd.cut(y[target], bins=bins, labels=labels)
    
    return y_class

def binary_transform_with_weight(y, threshold=1000, uncertainty=0.5, targets=None):
    """
    Transforms continuous values into binary classes and calculates 
    sample weights based on distance from threshold and data uncertainty.
    """
    if targets is None:
        targets = y.columns

    y_class = pd.DataFrame(index=y.index)
    certainty_multiplier = 1.0 - uncertainty
    gray_area_width = threshold * uncertainty

    for target in targets:
        
        y_class[target] = pd.cut(
            y[target],
            bins=[-np.inf, threshold, np.inf],
            labels=['LC', 'HC']
        )

        # 2. Calculate Distance Weight (Linear Ramp)
        # Distance from threshold (e.g., |750 - 1000| = 250)
        raw_dist = np.abs(y[target] - threshold)
        
        # Scale by gray_area_width (e.g., 250 / 500 = 0.5 weight)
        # Use .clip(0, 2) to ensure anything > 1500 or < 500 stays at 1.0 after multiplying by certainty_multiplier
        dist_weight = (raw_dist / gray_area_width).clip(0, 2)
        y_class[target + '_weight'] = dist_weight * certainty_multiplier

    return y_class

def binary_transform(y, threshold = 1000, targets=['DON', 'Total_DON']):
    bins = [-float('inf'), threshold, float('inf')]
    labels = ['LC', 'HC']
    
    y_class = pd.DataFrame(index=y.index)
    
    for target in targets:
        y_class[target] = pd.cut(y[target], bins=bins, labels=labels)
    
    return y_class
