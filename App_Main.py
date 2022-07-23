from KNN_Neighbors_Optimal import KNN_Optimal
from Output_Training_Test_Network import output_network
from sklearn import datasets, model_selection, linear_model, multiclass, neural_network, svm, tree, naive_bayes, neighbors, ensemble


# ===================================== Step 1: Load Data ============================================
iris = datasets.load_iris()
X = iris.data
Labels = iris.target
# ===================================== Step 2: Normalize =============================================

# ===================================== Step 3: Feature Extraction ====================================
# ===================================== Step 4: Classification ========================================
TypeClass = 'AdaBoost'  # LogisticRegression, MLP, K_NN, SVM, DT, NB, RandomForest,AdaBoost
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
    Data_Train, Data_Test, Label_Train, Label_Test = model_selection.train_test_split(X, Labels, test_size=0.3, random_state=1)
    Num_K = KNN_Optimal(Data_Train, Label_Train, Data_Test, Label_Test, N=21)  # Obtain optimal K
    model = neighbors.KNeighborsClassifier(n_neighbors=Num_K, metric='minkowski')  # Train Network
elif TypeClass == 'RandomForest':
    model = ensemble.RandomForestClassifier(n_estimators=100, max_depth=2, criterion='entropy', random_state=0)
elif TypeClass == 'AdaBoost':
    model = ensemble.AdaBoostClassifier(n_estimators=100, random_state=0)

Accuracy_Train, Cr_Train, Accuracy_Test, Cr_Test = output_network(X, Labels, model, TypeClass, K_fold=5)
