import numpy as np
import matplotlib.pyplot as plt
from Plot_decision_regions import plot_decision_regions


def plot_classifier(data_train, label_train, data_test, label_test, label_predict_train, label_predict_test, k_fold,  model, type_class):
    if np.shape(data_train)[1] == 2:
        plot_decision_regions(data_train, label_train, data_test, label_test, model, type_class)
    elif np.shape(data_train)[1] == 3:
        lab_tr = np.unique(label_train)
        fig = plt.figure(figsize=(8, 5))
        ax1 = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122, projection='3d')

        for i in range(0, len(lab_tr)):
            ax1.plot3D(data_train[label_predict_train == lab_tr[i], 0], data_train[label_predict_train == lab_tr[i], 1], data_train[label_predict_train == lab_tr[i], 2], '.')
            ax2.plot3D(data_test[label_predict_test == lab_tr[i], 0], data_test[label_predict_test == lab_tr[i], 1], data_test[label_predict_test == lab_tr[i], 2], '.')
        ax1.title.set_text('Training'), ax1.set_xlabel('Feature 1'), ax1.set_ylabel('Feature 2'), ax1.set_zlabel('Feature 3'),
        ax1.legend(lab_tr, title='Class', ncol=3, handlelength=0.8, handletextpad=0.2), ax2.legend(lab_tr, title='Class', ncol=3, handlelength=0.8, handletextpad=0.2)
        ax1.view_init(30, 60), ax2.view_init(30, 60), ax2.title.set_text('Test'), ax2.set_xlabel('Feature 1'), ax2.set_ylabel('Feature 2'), ax2.set_zlabel('Feature 3'),
        pos1 = ax2.get_position()
        fig.suptitle('Classification Type: ' + type_class+'; '+str(k_fold)+'th K-fold:', fontsize=12, y=pos1.x1), fig.show()
