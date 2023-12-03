import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def plot_decision_regions(data_train, label_train, data_test, label_test, model, type_class):

    resolution = 0.03
    lab_tr = np.unique(label_train)
    lab_te = np.unique(label_test)

    x_combined = np.vstack((data_train, data_test))
    y_combined = np.hstack((label_train, label_test))
    x1_min, x1_max = x_combined[:, 0].min() - 1, x_combined[:, 0].max() + 1
    x2_min, x2_max = x_combined[:, 1].min() - 1, x_combined[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    z = model.predict(np.array([xx1.ravel(), xx2.ravel()]).T)

    # --------------------------------------------------- Result Plot ---------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey='row', figsize=(8, 5))
    fig.suptitle('Classification Type: ' + type_class, fontsize=12)
    colors = list(reversed(sns.color_palette("bright", len(lab_tr)).as_hex()))
    cmp = ListedColormap(colors[:len(lab_tr)])
    ax1.contourf(xx1, xx2, z.reshape(xx1.shape), alpha=0.35, cmap=cmp)
    ax1.set_xlim(xx1.min(), xx1.max())
    ax1.set_ylim(xx2.min(), xx2.max())

    ax2.contourf(xx1, xx2, z.reshape(xx1.shape), alpha=0.35, cmap=cmp)
    ax2.set_xlim(xx1.min(), xx1.max())
    ax2.set_ylim(xx2.min(), xx2.max())

    for i in range(0, len(np.unique(y_combined))):
        ax1.plot(data_train[label_train == lab_tr[i], 0], data_train[label_train == lab_tr[i], 1], '.', color=colors[i], label=lab_tr[i], markersize=10)
        ax2.plot(data_test[label_test == lab_te[i], 0], data_test[label_test == lab_te[i], 1], '.', color=colors[i], label=lab_tr[i], markersize=10)
        
    ax1.title.set_text('Training'), ax2.title.set_text('Test'), ax1.legend(title='Class', ncol=3, handlelength=0.8, handletextpad=0.2)
    ax2.legend(title='Class', ncol=3, handlelength=0.8, handletextpad=0.2), ax1.set(xlabel='Feature 1', ylabel='Feature 2'), ax2.set(xlabel='Feature 1')
    plt.tight_layout(), plt.show()

