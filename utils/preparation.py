import pandas as pd

## The utility to transform continueous DON values into three classes: Low Contamination (LC), Uncertain (UC), and High Contamination (HC).
def three_level_transform(y, targets=['DON', 'Total_DON']):
    bins = [-float('inf'), 500, 1500, float('inf')]
    labels = ['LC', 'UC', 'HC']
    
    y_class = pd.DataFrame(index=y.index)
    
    for target in targets:
        y_class[target] = pd.cut(y[target], bins=bins, labels=labels)
    
    return y_class

# transform into two classes: Low Contamination (LC) and High Contamination (HC)
def binary_transform(y, targets=['DON', 'Total_DON']):
    bins = [-float('inf'), 1000, float('inf')]
    labels = ['LC', 'HC']
    
    y_class = pd.DataFrame(index=y.index)
    
    for target in targets:
        y_class[target] = pd.cut(y[target], bins=bins, labels=labels)
    
    return y_class