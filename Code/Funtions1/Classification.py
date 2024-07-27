import numpy as np
from Roc_Curve import roc_curve
from KNN_Neighbors_Optimal import knn_optimal
from Data_Normalization import data_normalization
from Roc_Curve_all_kfold import roc_curve_all_kfold
from Plot_Classification import plot_classification
from sklearn import metrics, neighbors, model_selection


# ================================================== classification ====================================================== 
def classification(model, data, labels, fig, axs, k_fold=5, normalize_active="off", method="MinMaxScaler",  display_fold_classification=1, display_classification="off", display_normalize_classification="off", display_Roc_classes="off", display_all_kfold="off", display_Roc_classification="off",fig_size_Roc=(5, 3), fig_size_classification=(5, 3), display_optimal_k="off", type_class="LR"):
   # -------------------------------------------------------- K_fold --------------------------------------------------------------
   cv = model_selection.StratifiedKFold(n_splits=k_fold)     
   # -------------------------------------------------------- Parameters ----------------------------------------------------------
   tpr_tr = np.zeros((100, k_fold))
   tpr_te = np.zeros((100, k_fold))
   auc_tr = np.zeros(k_fold)
   auc_te = np.zeros(k_fold)
   accuracy_tr, accuracy_te, f1_tr, f1_te, precision_tr, precision_te, recall_tr, recall_te = [], [], [], [], [], [],[], []
   
   for j, (train, test) in enumerate(cv.split(data, labels)):  
      # --------------------------------------------------- Split data ------------------------------------------------------------
      data_train = data[train] 
      data_test = data[test]
      label_train = labels[train]
      label_test = labels[test]
      # -------------------------------------------------- Data normalization -----------------------------------------------------
      if normalize_active == "on": 
          data_train, data_test = data_normalization(data_train, data_test, method=method)   # method 1: MinMaxScaler, method 2: StandardScaler
      # ------------------------------------------------------ KNN methods --------------------------------------------------------
      if type_class == "KNN":
            num_k = knn_optimal(data_train, label_train, data_test, label_test, display_optimal_k, n=21, fig_size=(3.5,2.5))  # Obtain optimal K
            model = neighbors.KNeighborsClassifier(n_neighbors=num_k, weights='uniform', metric='minkowski')
      # ---------------------------------------------------- Training Network -----------------------------------------------------
      model.fit(data_train, label_train)                                           # Fit i.e., Training
      # ---------------------------------------------- Training Section -----------------------------------------------------------
      label_predict_train = model.predict(data_train)
      accuracy_tr.append(metrics.accuracy_score(label_train, label_predict_train)) # Calculate accuracy  
      f1_tr.append(metrics.f1_score(label_train, label_predict_train, average='weighted')) # Calculate F1 score
      precision_tr.append(metrics.precision_score(label_train, label_predict_train, average='weighted'))  # Calculate precision
      recall_tr.append(metrics.recall_score(label_train, label_predict_train, average='weighted'))   # Calculate recall
      # ----------------------------------------------- Test Section --------------------------------------------------------------
      label_predict_test = model.predict(data_test)
      accuracy_te.append(metrics.accuracy_score(label_test, label_predict_test)) # Calculate accuracy  
      f1_te.append(metrics.f1_score(label_test, label_predict_test, average='weighted')) # Calculate F1 score
      precision_te.append(metrics.precision_score(label_test, label_predict_test, average='weighted'))  # Calculate precision
      recall_te.append(metrics.recall_score(label_test, label_predict_test, average='weighted'))   # Calculate recall
      # cr_test = metrics.classification_report(label_test, label_predict_test, labels=np.unique(label_test))
      # --------------------------------------------- Compute ROC curve and area the curve ----------------------------------------
      tpr_tr[:, j], tpr_te[:, j], auc_tr[j], auc_te[j] = roc_curve(model, data_train, data_test, label_train, label_test, j, type_class,display_Roc_classes, fig_size_Roc)
      
      if display_fold_classification == j:
          plot_classification(data_train, label_train, data_test, label_test, model, j, type_class, display_classification, display_normalize_classification, fig_size_classification)

   roc_curve_all_kfold(tpr_tr, tpr_te, auc_tr, auc_te, fig, axs, k_fold, type_class, display_all_kfold, display_Roc_classification, fig_size_Roc)
   
   return accuracy_tr, accuracy_te, f1_tr, f1_te, precision_tr, precision_te, recall_tr, recall_te, type_class
