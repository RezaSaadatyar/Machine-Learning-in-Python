import numpy as np
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions


def plot_classifier(Data_Train, Label_Train, Data_Test, Label_Test, model, TypeClass):

    if np.shape(Data_Train)[1] == 2:
        fig, (ax1, ax2) = plt.subplots(1, 2, sharey='row', figsize=(8, 5))
        fig.suptitle('Classification Type: ' + TypeClass, fontsize=12)
        ax1.plot_decision_regions(Data_Train, Label_Train, clf=model, legend=3), plt.title('Training')
        ax2.plot_decision_regions(Data_Test, Label_Test, clf=model, legend=3), plt.title('Test')
        plt.show()
    else:
        lab_tr = np.unique(Label_Train)
        fig = plt.figure(figsize=(8, 5))
        fig.suptitle('Classification Type: ' + TypeClass, fontsize=12)
        ax1 = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122, projection='3d')

        for i in range(0, len(lab_tr)):
            ax1.plot3D(Data_Train[Label_Train == lab_tr[i], 0], Data_Train[Label_Train == lab_tr[i], 1], Data_Train[Label_Train == lab_tr[i], 2], '.')
            ax2.plot3D(Data_Test[Label_Test == lab_tr[i], 0], Data_Test[Label_Test == lab_tr[i], 1], Data_Test[Label_Test == lab_tr[i], 2], '.')
        ax1.title.set_text('Training'), ax1.set_xlabel('Feature 1'), ax1.set_ylabel('Feature 2'), ax1.set_zlabel('Feature 3'), ax1.legend(lab_tr)
        ax1.view_init(30, 60)
        ax2.title.set_text('Test'), ax2.set_xlabel('Feature 1'), ax2.set_ylabel('Feature 2'), ax2.set_zlabel('Feature 3'), ax2.legend(lab_tr)
        ax2.view_init(30, 60)
        fig.show()
