# ==========================================================================
# ============================ Machine Learning  ============================
# ====================== Presented by: Reza Saadatyar  =====================
# =================== E-mail: Reza.Saadatyar92@gmail.com  ==================
# ============================  2022-2023 ==================================
# The program will run automatically when you run code/file Main.py, and you do not need to run any of the other codes.

# ===================================== Importing the required Libraries =============================
from sklearn import datasets
from Filtering import filtering
from Clustering import clustering
from Preparing_data import preparing_data
from Normalize import normalize_data
from Funtions.Classification import classification
from Plot_clustering import plot_cluster
from Funtions.Feature_Selection import featureselection
from Funtions.Feature_Extraction import feature_extraction
from Output_Training_Test_Network import output_network
# ============================================Step 1: Preparing the data ==============================================
iris = datasets.load_iris()
Data = iris.data
Labels = iris.target
# Data, Labels = datasets.make_blobs(150, 4, centers=4, random_state=0)
data, Labels = preparing_data(Data, Labels)
# ======================================== Step 2: Filtering & Data scaling ============================================
# Data = filtering(Data, F_low=5, F_high=10, Order=3, Fs=50, btype='bandpass')      # btype:'low', 'high', 'bandpass', 'bandstop'
# Data = normalize_data(Data, Type_Normalize='MinMaxScaler', Display_Figure='on')   # Type_Normalize:'MinMaxScaler', 'normalize'
# ======================================= Step 3: Feature Extraction & Selection =======================================
# Data = feature_extraction(Data, Labels, number_feature=3, number_neighbors=70, type_feature='LDA')
# Data = featureselection(Data, Labels, threshold=0.1, number_feature=3, c_l1fs=0.01, n_estimators_tfs=100,  type_feature='MI')
"""
Feature Extraction:
PCA:Principal Component Analysis; LDA:Linear discriminant analysis; ICA: Independent component analysis; SVD: Singular value decomposition
TSNE:T-distributed stochastic neighbor embedding; FA: Factor analysis; Isomap: Isometric Feature Mapping
Feature Selection:
Variance; Mutual information (MI); Chi-square test (Chi-square); fisher_score (FS); Forward feature selection (FFS);
Backward feature selection (BFS); Exhaustive Feature Selection (EFS); Recursive feature elimination (RFE); Random Forest (RF)
Univariate feature selection (UFS); L1-based feature selection (L1FS), Tree-based feature selection (TFS)
"""
# ===================================== Step 4: Classification & Clustering ==========================================
# ----------------------------------------- Step 4: Classification ---------------------------------------------------
model, type_class = classification(Data, Labels, type_class='XGBoost', hidden_layer_mlp=(10,), max_iter=200, kernel_svm='rbf',
                               c_svm=10, gamma_svm=0.7, max_depth=5, criterion_dt='entropy', n_estimators=500)
Accuracy_Train, Cr_Train, Accuracy_Test, Cr_Test = output_network(Data, Labels, model, type_class, k_fold=5)
"""
type_class: 'KNN', 'LR', 'MLP', 'SVM', 'DT', 'NB', 'RF', 'AdaBoost', 'XGBoost', 'LDA'
LR: LogisticRegression; MLP: Multilayer perceptron, SVM:Support Vector Machine; DT: Decision Tree; NB: Naive Bayes;
RF: Random Forest; AdaBoost; XGBoost; LDA: Linear Discriminant Analysis; KNN:K-Nearest Neighbors 
Parameters:
The number of hidden layers: hidden_layer_mlp; The number of epochs MLP: max_iter,
kernel_svm=‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’;  c_svm=Regularization parameter, 
gamma_svm=Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’. 
max_depth=The maximum depth of the tree, random forest and XGBoost; criterion= 'gini', 'entropy', 'log_loss';
n_estimators:The number of trees in the forest.
"""
# ----------------------------------------- Step 4: Clustering -----------------------------------------------------
clustering(Data, n_clusters=3, max_iter=100, thr_brich=0.5,  branchfactor_brich=50, n_neighbor_SpecCluster=10,
           minsamples_optics=15, max_dist_optics=5, batch_size_MBKmeans=10, type_cluster='MiniBatchKMeans')
"""
type_cluster: kmeans; Agglomerative; DBSCAN; GMM:Gaussian Mixture Models; Meanshift; Birch; SpectralClustering; 
OPTICS; MiniBatchKMeans
"""
