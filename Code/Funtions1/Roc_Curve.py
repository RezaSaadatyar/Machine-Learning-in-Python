import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

# ===================================================== Roc curve ======================================================== 
def roc_curve(model, data_train, data_test, label_train, label_test, k_fold, type_class, display_Roc_classes, fig_size_Roc=(5, 3)):
   # ----------------------- Initialize lists to store fpr, tpr, and roc_auc for each class ------------------------------
   fpr_tr, tpr_tr, fpr_te, tpr_te = dict(), dict(), dict(), dict()                                 
   roc_auc_tr, roc_auc_te = [], []
   mean_tpr_tr, mean_tpr_te = 0.0, 0.0
   mean_fpr = np.linspace(0, 1, 100)
   # ------------------------------------ Binarize the labels for each class ---------------------------------------------
   num_classes = np.max(label_train) + 1           
   label_tr = np.eye(num_classes)[label_train]  
   label_te = np.eye(num_classes)[label_test]  
   # ------------------------------------ Predict the labels for each class ---------------------------------------------
   y_scores_tr = model.predict_proba(data_train)   
   y_scores_te = model.predict_proba(data_test)   
   # --------------------------------------------- ROC curve for each class ---------------------------------------------
   for i in range(label_tr.shape[1]):            
      fpr_tr[i], tpr_tr[i], _ = metrics.roc_curve(label_tr[:, i], y_scores_tr[:, i])
      roc_auc_tr.append(metrics.auc(fpr_tr[i], tpr_tr[i]))
      mean_tpr_tr += np.interp(mean_fpr, fpr_tr[i], tpr_tr[i])  # Interpolate the mean_tpr at mean_fpr
      mean_tpr_tr[0] = 0.0
      
      fpr_te[i], tpr_te[i], _ = metrics.roc_curve(label_te[:, i], y_scores_te[:, i])
      roc_auc_te.append(metrics.auc(fpr_te[i], tpr_te[i]))
      mean_tpr_te += np.interp(mean_fpr, fpr_te[i], tpr_te[i])  # Interpolate the mean_tpr at mean_fpr
      mean_tpr_te[0] = 0.0
   # ------------------------ Calculate mean true positive rate across all folds and classes ----------------------------
   mean_tpr_tr = mean_tpr_tr/(i+1)                            
   mean_tpr_te = mean_tpr_te/(i+1)
   mean_auc_tr = metrics.auc(mean_fpr, mean_tpr_tr)
   mean_auc_te = metrics.auc(mean_fpr, mean_tpr_te)
   # ------------------------------------------- Plot ROC curve for each class ------------------------------------------
   if display_Roc_classes == "on":
      fig1, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=fig_size_Roc, constrained_layout=True)
      for i in range(label_tr.shape[1]):
         ax1.plot(fpr_tr[i], tpr_tr[i], lw=1.2, label=f"C{i}:{roc_auc_tr[i]:.2f}")
         ax2.plot(fpr_te[i], tpr_te[i], lw=1.2, label=f"C{i}:{roc_auc_te[i]:.2f}")
      # ----------------------------- # Compute macro-average ROC curve and ROC area ------------------------------------
      all_fpr_tr = np.unique(np.concatenate([fpr_tr[i] for i in range(label_tr.shape[1])]))# First aggregate all false positive rates
      all_fpr_te = np.unique(np.concatenate([fpr_te[i] for i in range(label_te.shape[1])]))
      avg_tpr_tr = np.zeros_like(all_fpr_tr)                                               # Then interpolate all ROC curves at this points
      avg_tpr_te = np.zeros_like(all_fpr_te)                                               
      for i in range(label_tr.shape[1]):
         avg_tpr_tr += np.interp(all_fpr_tr, fpr_tr[i], tpr_tr[i])
         avg_tpr_te += np.interp(all_fpr_te, fpr_te[i], tpr_te[i])
         
      avg_tpr_tr /= label_tr.shape[1]                                                      # Finally average it and compute AUC
      avg_tpr_te /= label_te.shape[1] 
      fpr_tr[i+1] = all_fpr_tr
      fpr_te[i+1] = all_fpr_te
      tpr_tr[i+1] = avg_tpr_tr
      tpr_te[i+1] = avg_tpr_te
      roc_auc_tr.append(metrics.auc(fpr_tr[i+1], tpr_tr[i+1]))
      roc_auc_te.append(metrics.auc(fpr_te[i+1], tpr_te[i+1]))
      # -------------------------------- # Compute micro-average ROC curve and ROC area ----------------------------------
      fpr_tr[i+2], tpr_tr[i+2], _ = metrics.roc_curve(label_tr.ravel(), y_scores_tr.ravel())
      roc_auc_tr.append(metrics.auc(fpr_tr[i+2], tpr_tr[i+2]))
      fpr_te[i+2], tpr_te[i+2], _ = metrics.roc_curve(label_te.ravel(), y_scores_te.ravel())
      roc_auc_te.append(metrics.auc(fpr_te[i+2], tpr_te[i+2]))
      # ------------------------------------ Plot Macro avg & Micro avg --------------------------------------------------
      ax1.plot([0, 1], [0, 1], linestyle='--', color='gray', lw=1.2)
      ax1.plot(fpr_tr[i+1], tpr_tr[i+1] , color='blue', linestyle='-', lw=1.2, label=f"Macro avg:{roc_auc_tr[i+1]:.2f}")
      ax1.plot(fpr_tr[i+2], tpr_tr[i+2] , color='g', linestyle='-', lw=1.2, label=f"Micro avg:{roc_auc_tr[i+2]:.2f}")
      ax1.axis(xmin=-0.03, xmax=1, ymin=-0.03, ymax=1.03)         # Set x-axis and y-axis limits in a single line
      ax1.grid(True, linestyle='--', which='major', color='grey', alpha=0.5, axis="y")
      ax1.legend(title="AUC", loc='lower right',fontsize=9, ncol=1, frameon=True, labelcolor='linecolor', handlelength=0)
      ax1.set_xlabel('False Positive Rate (FPR)', fontsize=10), ax1.set_ylabel('True Positive Rate (TPR)', fontsize=10)

      ax2.plot([0, 1], [0, 1], linestyle='--', color='gray', lw=1.2)
      ax2.plot(fpr_te[i+1], tpr_te[i+1] , color='blue', linestyle='-', lw=1.2, label=f"Macro avg:{roc_auc_te[i+1]:.2f}")
      ax2.plot(fpr_te[i+2], tpr_te[i+2] , color='g', linestyle='-', lw=1.2, label=f"Micro avg:{roc_auc_te[i+2]:.2f}")
      ax2.axis(xmin=-0.03, xmax=1, ymin=-0.03, ymax=1.03)         # Set x-axis and y-axis limits in a single line
      ax2.set_xlabel('False Positive Rate (FPR)', fontsize=10)
      ax2.grid(True, linestyle='--', which='major', color='grey', alpha=0.5, axis="y")
      ax2.legend(title="AUC", loc='lower right',fontsize=9, ncol=1, frameon=True, labelcolor='linecolor', handlelength=0)
       
      plt.autoscale(axis="x", tight=True, enable=True)
      fig1.suptitle(f"{type_class} ROC Curve for a {k_fold+1}_fold cross-validation", fontsize=10)
        
   return mean_tpr_tr, mean_tpr_te, mean_auc_tr, mean_auc_te 
