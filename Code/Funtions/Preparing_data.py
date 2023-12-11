import numpy as np
from sklearn import preprocessing

# =============================================== Preparing data =========================================================
def preparing_data(data, labels):
    
    if data.shape[0] < data.shape[1]:
        data = data.T
    
    if len(labels.shape) > 1:
        labels = np.max(labels) + 1  # Assuming classes are 0-indexed
    
    mod = preprocessing.LabelEncoder()
    labels = mod.fit_transform(labels)
    
    return data, labels