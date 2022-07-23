import numpy as np
from sklearn import datasets, preprocessing
import matplotlib.pyplot as plt

# ===================================== Step 1: Load Data ==========================================
iris = datasets.load_iris()
Data = iris.data
Labels = iris.target

# =================================== Step 2: Normalize Methods ====================================
Type_Normalize = 'normalize'                    # 'MinMaxScaler', 'normalize',
def Normalize_Data(Data, Type_Normalize):
    if Type_Normalize == 'MinMaxScaler':
        Normalize = preprocessing.MinMaxScaler()
        Normalize.fit(Data)
        Min_Data = Normalize.data_min_
        Max_Data = Normalize.data_max_
        Normalize_Data = Normalize.transform(Data)  # (Data-min)/(max-min)
    elif Type_Normalize == 'normalize':
        Normalize_Data = preprocessing.normalize(Data, norm='l1', axis=0)   # l1, l2

    return Normalize_Data


Normalize_Data = Normalize_Data(Data, Type_Normalize)
# ======================================== Step 3: Plot ===========================================
plt.subplot(121)
plt.plot(Data[:, 0], Data[:, 1], '.', label='Raw Data'), plt.legend()
plt.subplot(122)
plt.plot(Normalize_Data[:, 0], Normalize_Data[:, 1], '.', label='Normalized Data'), plt.legend()
plt.show()
