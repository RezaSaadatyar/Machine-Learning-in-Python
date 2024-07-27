import numpy as np
import matplotlib.pyplot as plt

# =================================================== Performance ======================================================== 
def performance(accuracy_tr, accuracy_te, precision_tr, precision_te, f1_tr, f1_te, recall_tr, recall_te, type_method, k_fold, fig_size=(7, 3)):
    
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=fig_size, sharey="row", constrained_layout=True)

    mean_tr = [np.mean(accuracy_tr, axis=1), np.mean(f1_tr, axis=1), np.mean(precision_tr, axis=1), np.mean(recall_tr, axis=1)]
    std_tr = [np.std(accuracy_tr, axis=1), np.std(f1_tr, axis=1), np.std(precision_tr, axis=1), np.std(recall_tr, axis=1)]
    mean_te = [np.mean(accuracy_te, axis=1), np.mean(f1_te, axis=1), np.mean(precision_te, axis=1), np.mean(recall_te, axis=1)]
    std_te = [np.std(accuracy_te, axis=1), np.std(f1_te, axis=1), np.std(precision_te, axis=1), np.std(recall_te, axis=1)]

    bar_width = 1  # Change the bar width as needed
    x_label_distance = 3  # Adjust the distance between x-labels

    index = np.arange(len(type_method)) * (2*bar_width + x_label_distance)  # Create an array of x-values based on the bar width and distance

    for i, label in enumerate(['Accuracy', 'F1', 'Precision', 'Recall']):
        axs[0].bar(index + i * bar_width, mean_tr[i], bar_width, yerr=std_tr[i], capsize=2, label=label)
        axs[1].bar(index + i * bar_width, mean_te[i], bar_width, yerr=std_te[i], capsize=2, label=label)

    axs[0].set_title("Training", fontsize=10, pad=0, loc="left")
    axs[0].set_xticks(index + (4 - 1) * bar_width / 2, type_method)
    axs[0].grid(True, linestyle='--', which='major', color='grey', alpha=0.3, axis="y")
    axs[0].set_xlabel('Classifiers', fontsize=10, va='center'), axs[0].set_ylabel('Scores', fontsize=10, va='center')
    axs[0].legend(fontsize=9.5, loc="best", ncol=2, handlelength=0, handletextpad=0.25, frameon=True, labelcolor='linecolor') 
    axs[0].tick_params(axis='y', length=1.5, width=1, which="both", bottom=False, top=False, labelbottom=True, labeltop=True)
    axs[0].tick_params(axis='x', length=1.5, width=1, which='both', bottom=True, top=False, labelbottom=True, labeltop=False, pad=0, rotation=90)

    axs[1].set_title("Test", fontsize=10, pad=0, loc="right")
    axs[1].set_xticks(index + (4 - 1) * bar_width / 2, type_method)
    axs[1].grid(True, linestyle='--', which='major', color='grey', alpha=0.3, axis="y")
    fig.suptitle(f"Performance Metrics for Different Classifiers for a {k_fold}_fold cross-validation", fontsize=10)
    axs[1].set_xlabel('Classifiers', fontsize=10, va='center'), axs[0].set_ylabel('Scores', fontsize=10, va='center')
    axs[1].tick_params(axis='y', length=1.5, width=1, which="both", bottom=False, top=False, labelbottom=True, labeltop=True)
    axs[1].tick_params(axis='x', length=1.5, width=1, which='both', bottom=True, top=False, labelbottom=True, labeltop=False, pad=0, rotation=90)