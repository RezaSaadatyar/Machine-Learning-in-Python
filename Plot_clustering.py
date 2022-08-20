import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def plot_cluster(data, labels, type_cluster):
    lab_tr = np.unique(labels)
    fig = plt.figure(figsize=(7, 4))
    colors = list(reversed(sns.color_palette("bright", len(lab_tr)).as_hex()))
    if np.shape(data)[1] == 2:
        for i in range(0, len(lab_tr)):
            plt.plot(data[labels == lab_tr[i], 0], data[labels == lab_tr[i], 1], '.', color=colors[i], label=lab_tr[i], markersize=10)
        plt.legend(title='Class', ncol=3, handlelength=0.8, handletextpad=0.2), plt.xlabel('Feature 1'), plt.ylabel('Feature 2')
    elif np.shape(data)[1] == 3:
        ax = plt.axes(projection='3d')
        for i in range(0, len(lab_tr)):
            ax.plot3D(data[labels == lab_tr[i], 0], data[labels == lab_tr[i], 1], data[labels == lab_tr[i], 2], '.', color=colors[i], markersize=10)
        ax.set_xlabel('Feature 1'), ax.set_ylabel('Feature 2'), ax.set_zlabel('Feature 3'),
        ax.legend(lab_tr, title='Class', ncol=3, handlelength=0.8, handletextpad=0.2), ax.view_init(20, 80)
    plt.title('Classification Type: ' + type_cluster, fontsize=12), plt.tight_layout(), plt.show()
