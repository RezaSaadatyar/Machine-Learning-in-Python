import numpy as np
from sklearn import metrics, preprocessing

def Roc_curve(ytest, y_pred):

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    Classes = np.unique(ytest)
    ytest = preprocessing.label_binarize(ytest, classes=Classes)

    for i in range(len(Classes)):
        fpr[i], tpr[i], _ = metrics.roc_curve(ytest[:, i], y_pred[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(Classes))]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(len(Classes)):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally, average it and compute AUC
    mean_tpr /= len(Classes)
    tpr = [0]
    tpr.extend(mean_tpr)
    fpr = [0]
    fpr.extend(all_fpr)
    roc_auc = metrics.auc(all_fpr, mean_tpr)
    return fpr, tpr, roc_auc
