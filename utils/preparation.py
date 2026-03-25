import pandas as pd

## The utility to transform continueous DON values into three classes: Low Contamination (LC), Uncertain (UC), and High Contamination (HC).
def three_level_transform(y, targets=['DON', 'Total_DON']):
    bins = [-float('inf'), 500, 1500, float('inf')]
    labels = ['LC', 'UC', 'HC']
    
    y_class = pd.DataFrame(index=y.index)
    
    for target in targets:
        y_class[target] = pd.cut(y[target], bins=bins, labels=labels)
    
    return y_class

def binary_transform_with_weight(y, threshold=1000, targets=None):
    if targets is None:
        targets = y.columns

    y_class = pd.DataFrame(index=y.index)

    for target in targets:
        # class label
        y_class[target] = pd.cut(
            y[target],
            [-float('inf'), threshold, float('inf')],
            labels=['LC', 'HC']
        )
        # weight
        y_class[target + '_weight'] = pd.cut(
            y[target],
            [-float('inf'), threshold*0.5, threshold*1.5, float('inf')],
            labels=[1.0, 0.2, 1.0],
            ordered=False
        ).astype(float)

    return y_class

def binary_transform(y, threshold = 1000, targets=['DON', 'Total_DON']):
    bins = [-float('inf'), threshold, float('inf')]
    labels = ['LC', 'HC']
    
    y_class = pd.DataFrame(index=y.index)
    
    for target in targets:
        y_class[target] = pd.cut(y[target], bins=bins, labels=labels)
    
    return y_class
