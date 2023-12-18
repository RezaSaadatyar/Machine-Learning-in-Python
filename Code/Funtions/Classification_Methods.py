import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from Performance import performance
from Classification import classification
from sklearn import linear_model, neural_network, svm, tree, naive_bayes, ensemble, discriminant_analysis

# ============================================== Classification methods ================================================== 
def classification_methods(data, labels, k_fold, max_iter, solver_LR, hidden_layer_MLP, lr_MLP, kernel_SVM, C_SVM, criterion, lr_AdaBoost, lr_XGBoost, n_estimators, max_depth, solver_LDA, LR, MLP, SVM, DT, NB, RF, AdaBoost, XGBoost, LDA, KNN,
                           normalize_active, method, display_fold_classification, display_classification, display_normalize_classification, display_Roc_classes, display_all_kfold, display_Roc_classification, fig_size_Roc, fig_size_classification, fig_size_performance, display_optimal_k):
                           
   accuracy_tr, accuracy_te, f1_tr, f1_te, precision_tr, precision_te = [], [], [], [], [], []
   recall_tr, recall_te, type_method = [], [], []
   
   if display_Roc_classification == "on":
      fig, axs = plt.subplots(nrows=1, ncols=2, figsize=fig_size_Roc, constrained_layout=True)
   else:
      fig, axs = [], []
   # ------------------------------------------------------ Logistic Regression -------------------------------------------------------
   if LR == "on":
      model = linear_model.LogisticRegression(C=1, max_iter=max_iter, solver=solver_LR, penalty='l2', multi_class="multinomial", verbose=0)
      
      output = classification(model, data, labels, fig, axs, k_fold, normalize_active, method, display_fold_classification, display_classification, display_normalize_classification, display_Roc_classes, display_all_kfold, display_Roc_classification, fig_size_Roc, fig_size_classification, display_optimal_k, type_class="LR")

      accuracy_tr.append(output[0]), accuracy_te.append(output[1]), f1_tr.append(output[2]), f1_te.append(output[3]), 
      precision_tr.append(output[4]), precision_te.append(output[5]), recall_tr.append(output[6]), recall_te.append(output[7]),
      type_method.append(output[8]),
   # ----------------------------------------------------------- MLP ------------------------------------------------------------------
   if MLP == "on":
      model = neural_network.MLPClassifier(hidden_layer_sizes=hidden_layer_MLP, max_iter=500, alpha=1e-4, learning_rate='invscaling', solver='adam',
                                       random_state=1, learning_rate_init=lr_MLP, verbose=False , tol=1e-4)    

      output = classification(model, data, labels, fig, axs, k_fold, normalize_active, method, display_fold_classification, display_classification, display_normalize_classification, display_Roc_classes, display_all_kfold, display_Roc_classification, fig_size_Roc, fig_size_classification, display_optimal_k, type_class="MLP")

      accuracy_tr.append(output[0]), accuracy_te.append(output[1]), f1_tr.append(output[2]), f1_te.append(output[3]), 
      precision_tr.append(output[4]), precision_te.append(output[5]), recall_tr.append(output[6]), recall_te.append(output[7]),
      type_method.append(output[8]),
   # ------------------------------------------------------------- SVM ----------------------------------------------------------------
   if SVM == "on":
      model = svm.SVC(kernel=kernel_SVM, random_state=0, C=C_SVM, gamma="auto", probability=True) 
     
      output = classification(model, data, labels, fig, axs, k_fold, normalize_active, method, display_fold_classification, display_classification, display_normalize_classification, display_Roc_classes, display_all_kfold, display_Roc_classification, fig_size_Roc, fig_size_classification, display_optimal_k, type_class="SVM")

      accuracy_tr.append(output[0]), accuracy_te.append(output[1]), f1_tr.append(output[2]), f1_te.append(output[3]), 
      precision_tr.append(output[4]), precision_te.append(output[5]), recall_tr.append(output[6]), recall_te.append(output[7]),
      type_method.append(output[8]),
   # -------------------------------------------------------------- DT ------------------------------------------------------------------
   if DT == "on":
      model = tree.DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, random_state=0)

      output = classification(model, data, labels, fig, axs, k_fold, normalize_active, method, display_fold_classification, display_classification, display_normalize_classification, display_Roc_classes, display_all_kfold, display_Roc_classification, fig_size_Roc, fig_size_classification, display_optimal_k, type_class="DT")

      accuracy_tr.append(output[0]), accuracy_te.append(output[1]), f1_tr.append(output[2]), f1_te.append(output[3]), 
      precision_tr.append(output[4]), precision_te.append(output[5]), recall_tr.append(output[6]), recall_te.append(output[7]),
      type_method.append(output[8])
   # -------------------------------------------------------------- NB ------------------------------------------------------------------
   if NB == "on":
      model = naive_bayes.GaussianNB()

      output = classification(model, data, labels, fig, axs, k_fold, normalize_active, method, display_fold_classification, display_classification, display_normalize_classification, display_Roc_classes, display_all_kfold, display_Roc_classification, fig_size_Roc, fig_size_classification, display_optimal_k, type_class="NB")

      accuracy_tr.append(output[0]), accuracy_te.append(output[1]), f1_tr.append(output[2]), f1_te.append(output[3]), 
      precision_tr.append(output[4]), precision_te.append(output[5]), recall_tr.append(output[6]), recall_te.append(output[7]),
      type_method.append(output[8])
   # -------------------------------------------------------------- RF ------------------------------------------------------------------
   if RF == "on":
      model = ensemble.RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, criterion="gini", random_state=0)

      output = classification(model, data, labels, fig, axs, k_fold, normalize_active, method, display_fold_classification, display_classification, display_normalize_classification, display_Roc_classes, display_all_kfold, display_Roc_classification, fig_size_Roc, fig_size_classification, display_optimal_k, type_class="RF")

      accuracy_tr.append(output[0]), accuracy_te.append(output[1]), f1_tr.append(output[2]), f1_te.append(output[3]), 
      precision_tr.append(output[4]), precision_te.append(output[5]), recall_tr.append(output[6]), recall_te.append(output[7]),
      type_method.append(output[8])
   # ----------------------------------------------------------- AdaBoost ---------------------------------------------------------------
   if AdaBoost == "on":
      #  # model = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
      # # model = ensemble.AdaBoostClassifier(base_estimator=model, n_estimators=100, random_state=0)
      model = ensemble.AdaBoostClassifier(n_estimators=n_estimators, learning_rate=lr_AdaBoost, random_state=0)

      output = classification(model, data, labels, fig, axs, k_fold, normalize_active, method, display_fold_classification, display_classification, display_normalize_classification, display_Roc_classes, display_all_kfold, display_Roc_classification, fig_size_Roc, fig_size_classification, display_optimal_k, type_class="AdaBoost")

      accuracy_tr.append(output[0]), accuracy_te.append(output[1]), f1_tr.append(output[2]), f1_te.append(output[3]), 
      precision_tr.append(output[4]), precision_te.append(output[5]), recall_tr.append(output[6]), recall_te.append(output[7]),
      type_method.append(output[8])
   # ------------------------------------------------------------ XGBoost ---------------------------------------------------------------
   if XGBoost == "on":
      model = XGBClassifier(max_depth=5, n_estimators=n_estimators, learning_rate=lr_XGBoost, random_state=0, objective='multi:softpr_teob')

      output = classification(model, data, labels, fig, axs, k_fold, normalize_active, method, display_fold_classification, display_classification, display_normalize_classification, display_Roc_classes, display_all_kfold, display_Roc_classification, fig_size_Roc, fig_size_classification, display_optimal_k, type_class="XGBoost")

      accuracy_tr.append(output[0]), accuracy_te.append(output[1]), f1_tr.append(output[2]), f1_te.append(output[3]), 
      precision_tr.append(output[4]), precision_te.append(output[5]), recall_tr.append(output[6]), recall_te.append(output[7]),
      type_method.append(output[8])
   # -------------------------------------------------------------- LDA -----------------------------------------------------------------
   if LDA == "on":
      model = discriminant_analysis.LinearDiscriminantAnalysis(n_components=len(np.unique(labels)) - 1, solver=solver_LDA)

      output = classification(model, data, labels, fig, axs, k_fold, normalize_active, method, display_fold_classification, display_classification, display_normalize_classification, display_Roc_classes, display_all_kfold, display_Roc_classification, fig_size_Roc,fig_size_classification, display_optimal_k, type_class="LDA")

      accuracy_tr.append(output[0]), accuracy_te.append(output[1]), f1_tr.append(output[2]), f1_te.append(output[3]), 
      precision_tr.append(output[4]), precision_te.append(output[5]), recall_tr.append(output[6]), recall_te.append(output[7]),
      type_method.append(output[8])
   # -------------------------------------------------------------- KNN -----------------------------------------------------------------
   if KNN == "on":
      model = []

      output = classification(model, data, labels, fig, axs, k_fold, normalize_active, method, display_fold_classification, display_classification, display_normalize_classification, display_Roc_classes, display_all_kfold, display_Roc_classification, fig_size_Roc, fig_size_classification, display_optimal_k, type_class="KNN")

      accuracy_tr.append(output[0]), accuracy_te.append(output[1]), f1_tr.append(output[2]), f1_te.append(output[3]), 
      precision_tr.append(output[4]), precision_te.append(output[5]), recall_tr.append(output[6]), recall_te.append(output[7]),
      type_method.append(output[8]),

   performance(accuracy_tr, accuracy_te, precision_tr, precision_te, f1_tr, f1_te, recall_tr, recall_te, type_method, k_fold, fig_size_performance)