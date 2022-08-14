import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import svm, ensemble, feature_selection, linear_model
from mlxtend.feature_selection import SequentialFeatureSelector


def featureselection(Data, Labels, Threshold, Number_Feature, C_L1FS, N_estimators_TFS, Type_feature):
    # ----------------------------------- Filter Methods --------------------------------------------------------------
    if Type_feature == 'Variance':
        mod = feature_selection.VarianceThreshold(threshold=Threshold)
        mod.fit(Data)
        data = mod.transform(Data)
        mod.get_support()
    # ----------------------------------- Wrapper Methods --------------------------------------------------------------
    elif Type_feature == 'RFECV':    # Recursive feature elimination with cross validation
        mod = feature_selection.RFECV(estimator=linear_model.LogisticRegression(max_iter=1000), step=1, cv=5, scoring='accuracy', n_jobs=-1,
                                     )
        data = mod.fit(Data, Labels)
        mod.n_features_    # optimal number of feature
        accuracy = np.round(mod.cv_results_['mean_test_score'], 2)
        fig, ax = plt.subplots()
        ax.plot(np.linspace(1, len(accuracy), len(accuracy), endpoint=True, dtype='int'), accuracy)
        ax.set(xlabel="Number of features selected", ylabel="Cross validation score of number of selected features")
        ax.xaxis.set_ticks(np.linspace(1, len(mod.cv_results_['mean_test_score']), len(mod.cv_results_['mean_test_score']), endpoint=True, dtype='int'))
        ax.yaxis.set_ticks(accuracy)
        ax.set_title("Recursive feature elimination with cross validation", fontweight="bold", fontsize=10), plt.tight_layout()
        plt.show()
        data1 = mod.fit_transform(Data, Labels)
    elif Type_feature == 'RFE':
        mod = feature_selection.RFE(estimator=mod, n_features_to_select=Number_Feature, step=1)  # Use RFE to eliminate the less importance features
        data = mod.fit_transform(Data, Labels)
        mod.get_support()
        mod.ranking_

    # ----------------------------------- Embedded Methods --------------------------------------------------------------
    elif Type_feature == 'RF':
        mod = ensemble.RandomForestClassifier(n_estimators=10, random_state=0)
        mod.fit(Data, Labels)
        mod.feature_importances_     # Estimating features importance
        mod = feature_selection.SelectFromModel(mod, prefit=True, threshold=Threshold)
        data = mod.transform(Data)


    elif Type_feature == 'BFS':  # Backward feature selection
        mod = SequentialFeatureSelector(ensemble.RandomForestClassifier(n_jobs=-1), k_features=Number_Feature, forward=False, floating=False, cv=0,
                                        scoring='accuracy', verbose=2)
        mod.fit(Data, Labels)
        data = mod.transform(Data)
        df = pd.DataFrame.from_dict(mod.get_metric_dict()).T
        # df = df[["feature_idx", "avg_score"]]
        print(df)
        mod.k_feature_idx_
    elif Type_feature == 'FFS':  # forward feature selection
        mod = linear_model.LogisticRegression(max_iter=1000)
        mod = SequentialFeatureSelector(mod, k_features=Number_Feature, forward=True, cv=5, scoring='accuracy')
        mod.fit(Data, Labels)
        mod.k_feature_idx_       # Optimal number of feature

        data = mod.fit_transform(Data, Labels)
    elif Type_feature == 'UFS':  # univariate feature selection
        mod = feature_selection.SelectKBest(feature_selection.chi2, k=Number_Feature)
        data = mod.fit_transform(Data, Labels)
    elif Type_feature == 'L1FS':  # L1-based feature selection; The smaller C the fewer feature selected
        mod = svm.LinearSVC(C=C_L1FS, penalty='l1', dual=False)
        mod = mod.fit(Data, Labels)
        mod = feature_selection.SelectFromModel(mod, prefit=True)
        data = mod.transform(Data)
    elif Type_feature == 'TFS':  # Tree-based feature selection
        mod = ensemble.ExtraTreesClassifier(n_estimators=N_estimators_TFS)
        mod = mod.fit(Data, Labels)
        # mod.feature_importances_
        mod = feature_selection.SelectFromModel(mod, prefit=True)
        data = mod.transform(Data).toarray
    return data
