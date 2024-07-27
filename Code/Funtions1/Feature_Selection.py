import numpy as np
# from skfeature.function.similarity_based import fisher_score
from sklearn import preprocessing, svm, ensemble, feature_selection, linear_model
# from mlxtend.feature_selection import SequentialFeatureSelector, ExhaustiveFeatureSelector


def feature_selected(data, labels, number_feature, threshold_var, n_neighbors_MI, L1_Parameter, type_feature_selection):  
   # ----------------------------------- Filter Methods --------------------------------------------------------------
   if type_feature_selection == 'Variance':             # Variance
      mod = feature_selection.VarianceThreshold(threshold=threshold_var)
      mod.fit(data)
      data = mod.transform(data)
   elif type_feature_selection == 'MI':                 # Mutual information
      mod = feature_selection.mutual_info_classif(data, labels, n_neighbors=n_neighbors_MI)
      data = data[:, np.argsort(mod)[-number_feature:]]
   elif type_feature_selection == 'UFS':                # Univariate feature selection
      scaler = preprocessing.MinMaxScaler()             # Perform Min-Max scaling
      data = scaler.fit_transform(data)
      mod = feature_selection.SelectKBest(feature_selection.chi2, k=number_feature)
      data = mod.fit_transform(data, labels)
   # elif type_feature_selection == 'FS':               # Fisher_score
   #    mod = fisher_score.fisher_score(data, labels)
   #    data = data[:, mod[-number_feature:]]
   # ----------------------------------- Wrapper Methods --------------------------------------------------------------
   elif type_feature_selection == 'RFE':                # Recursive feature elimination
      mod = feature_selection.RFE(estimator=linear_model.LogisticRegression(max_iter=1000), n_features_to_select=number_feature)
      mod.fit(data, labels)
      data = mod.transform(data)
   # elif type_feature_selection == 'FFS':              # Forward feature selection
   #    mod = linear_model.LogisticRegression(max_iter=1000)
   #    mod = SequentialFeatureSelector(mod, k_features=number_feature, forward=True, cv=5, scoring='accuracy')
   #    mod.fit(data, labels)
   #    data = data[:, mod.k_feature_idx_]              # Optimal number of feature
   # elif type_feature_selection == 'EFS':              # Exhaustive Feature Selection
   #         mod = ExhaustiveFeatureSelector(estimator=linear_model.LogisticRegression(max_iter=1000), min_features=1, max_features=number_feature, scoring='accuracy', cv=3)
   #         mod.fit(data, labels)
   #         data = data[:, mod.best_idx_]
   # elif type_feature_selection == 'FFS':              # Forward feature selection
   #    mod = linear_model.LogisticRegression(max_iter=1000)
   #    mod = SequentialFeatureSelector(mod, k_features=number_feature, forward=True, cv=5, scoring='accuracy')
   #    mod.fit(input_data, labels)
   #    data = input_data[:, mod.k_feature_idx_]        # Optimal number of feature
   # ----------------------------------- Embedded Methods --------------------------------------------------------------
   elif type_feature_selection == 'RF':                 # Random forest
      mod = ensemble.RandomForestClassifier(n_estimators=10, random_state=0)
      mod.fit(data, labels)
      data = data[:, np.argsort(mod.feature_importances_)[-number_feature:]]
   elif type_feature_selection == 'L1FS':               # L1-based feature selection; The smaller C the fewer feature selected
      mod = svm.LinearSVC(C=L1_Parameter, penalty='l1', dual=False, max_iter=1000).fit(data, labels)
      mod = feature_selection.SelectFromModel(mod, prefit=True)
      data = mod.transform(data)
   elif type_feature_selection == 'TFS':                # Tree-based feature selection
      mod = ensemble.ExtraTreesClassifier(n_estimators=100)
      mod.fit(data, labels)
      data = data[:, np.argsort(mod.feature_importances_)[-number_feature:]]

   return data