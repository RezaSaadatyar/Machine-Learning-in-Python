import numpy as np
from sklearn import metrics, model_selection
import matplotlib.pyplot as plt
from Plot_Classifier import plot_classifier
from Roc_Curve import Roc_curve
from Plot_Roc_Curve import Plot_Roc_curve
from scipy import interp
from sklearn import metrics


def output_network(x, Labels, model, TypeClass, K_fold):
    if np.shape(x)[0] < np.shape(x)[1]:  # Convert Data training & Test >>>> m*n; m > n
        x = x.T
    # -------------------------------------------- K_fold ------------------------------------------
    cv = model_selection.StratifiedKFold(n_splits=K_fold)  # K-Fold
    # ------------------------------------------ ROC Parameters ------------------------------------
    i = 0
    tprs_tr = []
    aucs_tr = []
    tprs_te = []
    aucs_te = []
    accuracy_tr = []
    accuracy_te = []
    mean_fpr = np.linspace(0, 1, 50)

    # ----------------------------------------- Training Network ------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey='row', figsize=(8, 5))
    fig.suptitle('Classification Type: ' + TypeClass, fontsize=12)
    for train, test in cv.split(x, Labels):
        data_train = x[train]  # Part Training
        label_train = Labels[train]
        model.fit(data_train, label_train)  # Fit i.e., Training

        # ------------------------------------- Training Section ------------------------------------
        label_predict_train = model.predict(data_train)
        accuracy_train = model.score(data_train, label_train)
        cr_train = metrics.classification_report(label_train, label_predict_train, labels=np.unique(label_train))
        label_predict_train_prob = model.predict_proba(data_train)
        accuracy_tr.append(accuracy_train)
        # model.feature_importances_
        # -------------------------------------ROC for training -------------------------------------
        fpr_tr, tpr_tr, roc_auc_tr = Roc_curve(label_train, label_predict_train_prob)
        aucs_tr.append(roc_auc_tr)
        tprs_tr.append(interp(mean_fpr, fpr_tr, tpr_tr))
        tprs_tr[-1][0] = 0.0
        i += 1
        st = np.str(roc_auc_tr)
        ax1.plot(fpr_tr, tpr_tr, lw=1, alpha=0.4, label='ROC fold ' + np.str(i) + ' (AUC = ' + st[0:4] + ')')

        # ----------------------------------------- Test Section -------------------------------------
        data_test = x[test]
        label_test = Labels[test]
        label_predict_test = model.predict(data_test)
        accuracy_test = model.score(data_test, label_test)
        cr_test = metrics.classification_report(label_test, label_predict_test, labels=np.unique(label_test))
        label_predict_test_prob = model.predict_proba(data_test)
        accuracy_te.append(accuracy_test)

        # -------------------------------------- ROC for test -----------------------------------------
        fpr_te, tpr_te, roc_auc_te = Roc_curve(label_test, label_predict_test_prob)
        aucs_te.append(roc_auc_te)
        tprs_te.append(interp(mean_fpr, fpr_te, tpr_te))
        tprs_te[-1][0] = 0.0
        st = np.str(roc_auc_te)
        ax2.plot(fpr_te, tpr_te, lw=1, alpha=0.4, label='ROC fold ' + np.str(i) + ' (AUC = ' + st[0:4] + ')')

    ax1.set_ylabel("True Positive Rate")
    Plot_Roc_curve(tprs_tr, mean_fpr, aucs_tr, ax1, title='Training')
    Plot_Roc_curve(tprs_te, mean_fpr, aucs_te, ax2, title='Test')
    plt.show()
    plot_classifier(data_train, label_train, data_test, label_test, model, TypeClass)
    return np.mean(accuracy_tr), cr_train, np.mean(accuracy_te), cr_test

