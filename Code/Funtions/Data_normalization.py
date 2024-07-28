import numpy as np
from sklearn import preprocessing

# ========================================= Data normalization ===========================================
def data_normalization(x_train, x_test, method="minmax"):
    """
    Normalizes training and testing datasets according to the specified method.

    Parameters:
    x_train (array-like): Training data.
    x_test (array-like): Testing data.
    method (str): Normalization method - 'MinMaxScaler', 'StandardScaler', or 'MeanNormalization'.

    Returns:
    tuple: Normalized training and testing data arrays.
    """
    # Ensure the input is an array and reshape if 1D
    x_train = np.array(x_train).reshape(-1, 1) if x_train.ndim == 1 else np.array(x_train)
    x_test = np.array(x_test).reshape(-1, 1) if x_test.ndim == 1 else np.array(x_test)
    
    # Select the normalization method
    if method.lower() == "minmax":
        norm = preprocessing.MinMaxScaler()
    elif method.lower() == "standard":
        norm = preprocessing.StandardScaler()
    elif method.lower() == "meannormalize":
        # Implementing mean normalization manually
        def mean_normalize(data):
            mu = np.mean(data, axis=0)
            data_range = np.max(data, axis=0) - np.min(data, axis=0)
            return (data - mu) / data_range
        x_train = mean_normalize(x_train)
        x_test = mean_normalize(x_test)
        return x_train, x_test
    else:
        raise ValueError("Unsupported normalization method specified!")

    # Apply the normalization
    x_train = norm.fit_transform(x_train)
    x_test = norm.transform(x_test)
    
    return x_train, x_test
