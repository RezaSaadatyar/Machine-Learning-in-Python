import numpy as np
from xgboost import XGBClassifier
from KNN_Neighbors_Optimal import KNN_Optimal
from sklearn import model_selection, linear_model, multiclass, neural_network, svm, tree, naive_bayes, neighbors, ensemble, discriminant_analysis


def classification(data, labels, type_class, hidden_layer_mlp, max_iter, kernel_svm, c_svm, gamma_svm, max_depth, criterion_dt, n_estimators):
    if type_class == 'LR':
        model = multiclass.OneVsRestClassifier(linear_model.LogisticRegression(C=1e5, solver='lbfgs', multi_class='multinomial', penalty='l2'))  # sag, newton-c,lbfg;  penalty='l2', Nonr
    elif type_class == 'MLP':
        model = neural_network.MLPClassifier(hidden_layer_sizes=hidden_layer_mlp, max_iter=max_iter, alpha=1e-4, learning_rate='invscaling', solver='lbfgs',
                                             random_state=0, verbose=False, learning_rate_init=0.05)
    elif type_class == 'SVM':
        model = svm.SVC(kernel=kernel_svm, random_state=0, C=c_svm, gamma=gamma_svm, probability=True)  # kernel='rbf', 'poly', 'linear'
    elif type_class == 'DT':
        model = tree.DecisionTreeClassifier(criterion=criterion_dt, max_depth=max_depth, random_state=0)
    elif type_class == 'NB':
        model = naive_bayes.GaussianNB()
    elif type_class == 'RF':
        model = ensemble.RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, criterion=criterion_dt, random_state=0)
    elif type_class == 'AdaBoost':
        # model = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
        # model = ensemble.AdaBoostClassifier(base_estimator=model, n_estimators=100, random_state=0)
        model = ensemble.AdaBoostClassifier(n_estimators=n_estimators, learning_rate=1, random_state=0)
    elif type_class == 'XGBoost':
        model = XGBClassifier(max_depth=max_depth, n_estimators=n_estimators, learning_rate=0.01, random_state=0, objective='multi:softprob')
    elif type_class == 'LDA':
        model = discriminant_analysis.LinearDiscriminantAnalysis(n_components=len(np.unique(labels)) - 1)
    elif type_class == 'KNN':
        data_train, data_test, label_train, label_test = model_selection.train_test_split(data, labels, test_size=0.3, random_state=1)
        num_k = KNN_Optimal(data_train, label_train, data_test, label_test, N=21)  # Obtain optimal K
        model = neighbors.KNeighborsClassifier(n_neighbors=num_k, metric='minkowski')
    return model, type_class
