import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

def Plot_Roc_curve(tpr, mean_fpr, auc, ax, title):
    mean_tpr = np.mean(tpr, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = metrics.auc(mean_fpr, mean_tpr)
    std_auc = np.std(auc)
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4)
    ax.plot(mean_fpr, mean_tpr, label='Mean ROC (AUC = ' + np.str(np.round(mean_auc, 2)) + '$\pm$' +
                                      np.str(np.round(std_auc, 3)) + ')', lw=2, alpha=.8, color='b')
    std_tpr = np.std(tpr, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.25, label=r'$\pm$  std. dev.')
    ax.set_xlabel("False Positive Rate")
    ax.set_title(title)
    ax.legend(fontsize=8)



