import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skfeature.function.similarity_based import fisher_score
from sklearn import svm, ensemble, feature_selection, linear_model
from Plot_Feature_Extraction_Selection import plot_feature_extraction_selection
from mlxtend.feature_selection import SequentialFeatureSelector, ExhaustiveFeatureSelector


def featureselection(input_data, labels, threshold, number_feature, c_l1fs, n_estimators_tfs,  type_feature):
    # ----------------------------------- Filter Methods --------------------------------------------------------------
    if type_feature == 'Variance':
        mod = feature_selection.VarianceThreshold(threshold=threshold)
        mod.fit(input_data)
        data = mod.transform(input_data)
        mod.get_support()
    elif type_feature == 'MI':                 # Mutual information
        mod = feature_selection.mutual_info_classif(input_data, labels)
        data = input_data[:, np.argsort(mod)[-number_feature:]]
    elif type_feature == 'Chi-square':         # Chi-square Test
        mod = feature_selection.SelectKBest(feature_selection.chi2, k=number_feature)
        data = mod.fit_transform(input_data, labels)
    elif type_feature == 'FS':                 # Fisher_score
        mod = fisher_score.fisher_score(input_data, labels)
        data = input_data[:, mod[-number_feature:]]
    # ----------------------------------- Wrapper Methods --------------------------------------------------------------
    elif type_feature == 'FFS':                # Forward feature selection
        mod = linear_model.LogisticRegression(max_iter=1000)
        mod = SequentialFeatureSelector(mod, k_features=number_feature, forward=True, cv=5, scoring='accuracy')
        mod.fit(input_data, labels)
        data = input_data[:, mod.k_feature_idx_]     # Optimal number of feature
    elif type_feature == 'BFS':                # Backward feature selection
        mod = SequentialFeatureSelector(estimator=linear_model.LogisticRegression(max_iter=1000), k_features=number_feature, forward=False, floating=False, cv=0,
                                        scoring='accuracy', verbose=2)
        mod.fit(input_data, labels)
        data = mod.transform(input_data)
    elif type_feature == 'EFS':                # Exhaustive Feature Selection
        mod = ExhaustiveFeatureSelector(estimator=linear_model.LogisticRegression(max_iter=1000), min_features=1, max_features=number_feature, scoring='accuracy', cv=3)
        mod.fit(input_data, labels)
        data = input_data[:, mod.best_idx_]
    elif type_feature == 'RFE':                # Recursive feature elimination
        mod = feature_selection.RFE(estimator=linear_model.LogisticRegression(max_iter=1000), n_features_to_select=number_feature)
        mod.fit(input_data, labels)
        data = mod.transform(input_data)
    # ----------------------------------- Embedded Methods --------------------------------------------------------------
    elif type_feature == 'RF':                 # Random forest
        mod = ensemble.RandomForestClassifier(n_estimators=10, random_state=0)
        mod.fit(input_data, labels)
        data = input_data[:, np.argsort(mod.feature_importances_)[-number_feature:]]
    elif type_feature == 'UFS':                # Univariate feature selection
        mod = feature_selection.SelectKBest(feature_selection.chi2, k=number_feature)
        data = mod.fit_transform(input_data, labels)
    elif type_feature == 'L1FS':               # L1-based feature selection; The smaller C the fewer feature selected
        mod = svm.LinearSVC(C=c_l1fs, penalty='l1', dual=False, max_iter=1000).fit(input_data, labels)
        mod = feature_selection.SelectFromModel(mod, prefit=True)
        data = mod.transform(input_data)
    elif type_feature == 'TFS':                # Tree-based feature selection
        mod = ensemble.ExtraTreesClassifier(n_estimators=n_estimators_tfs)
        mod.fit(input_data, labels)
        data = input_data[:, np.argsort(mod.feature_importances_)[-number_feature:]]
    plot_feature_extraction_selection(input_data, data, labels, type_feature)
    return data
