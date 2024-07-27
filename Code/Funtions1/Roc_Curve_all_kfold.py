import numpy as np
import matplotlib.pyplot as plt

# ================================================= Display all kfold ==================================================== 
def roc_curve_all_kfold(tpr_tr, tpr_te, auc_tr, auc_te, fig, axs, k_fold, type_class, display_all_kfold="off", display_Roc_classification="off", fig_size_Roc=(4, 3)):

    mean_fpr = np.linspace(0, 1, 100)
    if display_Roc_classification == "on":
        axs[0].plot([0, 1], [0, 1], linestyle='--', color='gray', lw=1.2)
        axs[1].plot([0, 1], [0, 1], linestyle='--', color='gray', lw=1.2)
        axs[0].plot(mean_fpr, np.mean(tpr_tr, axis=1), linestyle='-', lw=1.2, label=f"AUC$_{{{type_class}}}$: {np.mean(auc_tr):.2f}")
        axs[1].plot(mean_fpr, np.mean(tpr_te, axis=1), linestyle='-', lw=1.2, label=f"AUC$_{{{type_class}}}$: {np.mean(auc_te):.2f}")
        axs[0].axis(xmin=-0.03, xmax=1, ymin=-0.03, ymax=1.03), axs[1].axis(xmin=-0.03, xmax=1, ymin=-0.03, ymax=1.03)        
        axs[0].grid(True, linestyle='--', which='major', color='grey', alpha=0.5, axis="y")
        axs[1].grid(True, linestyle='--', which='major', color='grey', alpha=0.5, axis="y")
        axs[0].set_xlabel('False Positive Rate (FPR)', fontsize=10), axs[1].set_xlabel('False Positive Rate (FPR)', fontsize=10)
        axs[0].set_ylabel('True Positive Rate (TPR)', fontsize=10)
        axs[0].legend(loc='lower right',fontsize=9, ncol=1, frameon=True, labelcolor='linecolor', handlelength=0)
        axs[1].legend(loc='lower right',fontsize=9, ncol=1, frameon=True, labelcolor='linecolor', handlelength=0)
        plt.autoscale(axis="x", tight=True, enable=True)
        fig.suptitle(f"ROC Curve for a {k_fold}_fold cross-validation", fontsize=10)

    if display_all_kfold=="on":
        fig1, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=fig_size_Roc, constrained_layout=True)
        ax1.plot([0, 1], [0, 1], linestyle='--', color='gray', lw=1.2)
        ax2.plot([0, 1], [0, 1], linestyle='--', color='gray', lw=1.2)
    
        for j in range(0, k_fold):
            
            ax1.plot(mean_fpr, tpr_tr[:, j], linestyle='-', lw=0.3, label=f"{j+1}-fold: {auc_tr[j]:.2f}")
            ax2.plot(mean_fpr, tpr_te[:, j], linestyle='-', lw=0.3, label=f"{j+1}-fold: {auc_te[j]:.2f}")

        tprs_upper_tr = np.minimum(np.mean(tpr_tr, axis=1) + np.std(tpr_tr, axis=1), 1)
        tprs_lower_tr = np.maximum(np.mean(tpr_tr, axis=1) - np.std(tpr_tr, axis=1), 0)
        tprs_upper_te = np.minimum(np.mean(tpr_te, axis=1) + np.std(tpr_te, axis=1), 1)
        tprs_lower_te = np.maximum(np.mean(tpr_te, axis=1) - np.std(tpr_te, axis=1), 0)
            
        ax1.fill_between(mean_fpr, tprs_lower_tr, tprs_upper_tr, color='grey', alpha=.2, label=r'$\pm$  std')
        ax1.plot(mean_fpr, np.mean(tpr_tr, axis=1), linestyle='-', lw=1.5, label=f"Mean: {np.mean(auc_tr):.2f}")
        ax2.fill_between(mean_fpr, tprs_lower_te, tprs_upper_te, color='grey', alpha=.2, label=r'$\pm$  std')
        ax2.plot(mean_fpr, np.mean(tpr_te, axis=1), linestyle='-', lw=1.5, label=f"Mean: {np.mean(auc_tr):.2f}")
            
        ax1.axis(xmin=-0.03, xmax=1, ymin=-0.03, ymax=1.03)         # Set x-axis and y-axis limits in a single line
        ax1.grid(True, linestyle='--', which='major', color='grey', alpha=0.5, axis="y")
        ax1.set_xlabel('False Positive Rate (FPR)', fontsize=10), ax1.set_ylabel('True Positive Rate (TPR)', fontsize=10)
        ax1.legend(title="AUC", loc='lower right',fontsize=9, ncol=1, frameon=True, labelcolor='linecolor', handlelength=0)

        ax2.axis(xmin=-0.03, xmax=1, ymin=-0.03, ymax=1.03)         # Set x-axis and y-axis limits in a single line
        ax2.set_xlabel('False Positive Rate (FPR)', fontsize=10)
        ax2.grid(True, linestyle='--', which='major', color='grey', alpha=0.5, axis="y")
        ax2.legend(title="AUC", loc='lower right',fontsize=9, ncol=1, frameon=True, labelcolor='linecolor', handlelength=0)
            
        plt.autoscale(axis="x", tight=True, enable=True)
        fig1.suptitle(f"{type_class} ROC Curve for a {k_fold}_fold cross-validation", fontsize=10)
    