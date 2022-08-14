# ===================================== Importing the required Libraries =============================
import matplotlib.pyplot as plt
import numpy as np
from KNN_Neighbors_Optimal import KNN_Optimal
from Filtering import filtering
from xgboost import XGBClassifier
from Normalize import normalize_data
from Feature_Selection import featureselection
from Feature_Extraction import feature_extraction
from Output_Training_Test_Network import output_network
from sklearn import preprocessing, datasets, model_selection, linear_model, multiclass, neural_network, svm, tree, naive_bayes, neighbors, ensemble, cluster, discriminant_analysis
from scipy.cluster import hierarchy
# ====================================Step 1: Preparing the data ======================================
iris = datasets.load_iris()
Data = iris.data
Labels = iris.target
#Data, Labels = datasets.make_blobs(150, 4, centers=4, random_state=0)
# Labels = preprocessing.LabelEncoder()
# Labels = Labels.fit_transform(Labels)
# ===================================== Step 2: Filtering & Data scaling =============================================
# Data = filtering(Data, F_low=5, F_high=10, Order=3, Fs=50, btype='bandpass')      # btype:'low', 'high', 'bandpass', 'bandstop'
# Data = normalize_data(Data, Type_Normalize='MinMaxScaler', Display_Figure='on')   # Type_Normalize:'MinMaxScaler', 'normalize'
# ==================================== Step 3: Feature Extraction & Selection ========================================
# Data = feature_extraction(Data, Labels, Number_Feature_PCA=3, Type_feature='ICA')     # Feature Extraction  ype_feature=LDA, PCA, ICA

Data = featureselection(Data, Labels, Threshold=0.1, Number_Feature=3, C_L1FS=0.01, N_estimators_TFS=100,  Type_feature='RFECV')
"""
Type_feature:UFS(univariate feature selection), L1FS, TFS, Variance, RF(Random Forest), BFS(Backward feature selection),
RFECV(Recursive feature elimination with cross validation),
forward: False i.e., direction is backward ; Number_Feature= integer or best   >> Sequential Forward/backward Selection
"""

"""
plot_feature_extraction_selection(input_data, output_feature, Labels, Type_feature)
# ===================================== Step 4: Classification ========================================
TypeClass = 'LDA'  # LogisticRegression, MLP, K_NN, SVM, DT, NB, RandomForest,AdaBoost, XGBoost, LDA
if TypeClass == 'LogisticRegression':
    model = multiclass.OneVsRestClassifier(linear_model.LogisticRegression(C=1e5, solver='lbfgs', multi_class='multinomial', penalty='l2'))  # sag, newton-c,lbfg;  penalty='l2', Nonr
elif TypeClass == 'MLP':
    model = neural_network.MLPClassifier(hidden_layer_sizes=(10,), max_iter=200, alpha=1e-4, learning_rate='invscaling',
                                         solver='lbfgs', random_state=5, verbose=False, learning_rate_init=0.05)
elif TypeClass == 'SVM':
    model = svm.SVC(kernel='rbf', random_state=0, C=10, gamma=0.7, probability=True)  # kernel='rbf', 'poly', 'linear'
elif TypeClass == 'DT':
    model = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
elif TypeClass == 'NB':
    model = naive_bayes.GaussianNB()
elif TypeClass == 'K_NN':
    Data_Train, Data_Test, Label_Train, Label_Test = model_selection.train_test_split(Data, Labels, test_size=0.3, random_state=1)
    Num_K = KNN_Optimal(Data_Train, Label_Train, Data_Test, Label_Test, N=21)  # Obtain optimal K
    model = neighbors.KNeighborsClassifier(n_neighbors=Num_K, metric='minkowski')  # Train Network
elif TypeClass == 'RandomForest':
    model = ensemble.RandomForestClassifier(n_estimators=100, max_depth=2, criterion='entropy', random_state=0)
elif TypeClass == 'AdaBoost':
    # model = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
    # model = ensemble.AdaBoostClassifier(base_estimator=model, n_estimators=100, random_state=0)
    model = ensemble.AdaBoostClassifier(n_estimators=100, learning_rate=1, random_state=0)
elif TypeClass == 'XGBoost':
    model = XGBClassifier(max_depth=3, n_estimators=100, learning_rate=0.01, random_state=0, objective='multi:softprob')
elif TypeClass == 'LDA':
    model = discriminant_analysis.LinearDiscriminantAnalysis(n_components=len(np.unique(Labels))-1)
Accuracy_Train, Cr_Train, Accuracy_Test, Cr_Test = output_network(Data, Labels, model, TypeClass, K_fold=5)

# ===================================== Step 4: Clustering ========================================
TypeCluster = 'DBSCAN'            # Kmeans, Agglomerative, DBSCAN
if TypeCluster == 'Kmeans':
    mod = cluster.KMeans(n_clusters=3, random_state=0)
    mod.cluster_centers_
    mod.fit(Data)
    y_predic = mod.predict(Data)
elif TypeCluster == 'Agglomerative':
    mod = cluster.AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')   # affinity: euclidean, manhattan; linkage: ward, single, average, complete
    mod.fit(Data)
    y_predic = mod.fit_predict(Data)
elif TypeCluster == 'DBSCAN':
    mod = cluster.DBSCAN(eps=1, min_samples=6)
    mod.fit(Data)
    y_predic = mod.fit_predict(Data)
print(y_predic)

hierarchy.dendrogram(hierarchy.linkage(Data, 'single'), labels=Labels)
plt.show()
"""







