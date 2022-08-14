import numpy as np
import seaborn as sns
from scipy import stats
from matplotlib import pyplot as plt


def plot_feature_extraction_selection(input_data, output_feature, Labels, Type_feature):
    if output_feature.shape[1] < 3:
        input_data = input_data[:, 0:2]
    lab_tr = np.unique(Labels)
    colors = np.array(sns.color_palette("bright", len(lab_tr)))
    for K in range(0, 2):
        fig = plt.figure(figsize=(6, 5))
        if K == 0:
            title = 'Original data'
            data = input_data
        else:
            title = 'Feature Extraction Type: ' + Type_feature
            data = output_feature

        if data.shape[1] < 3:
            ax = fig.add_axes((0.25, 0.78, 0.45, 0.14))
            ax2 = fig.add_axes((0.15, 0.1, 0.64, 0.65))
            ax3 = fig.add_axes((0.81, 0.2, 0.1, 0.47))
            for i in range(0, len(lab_tr)):
                _, bins = np.histogram(data[Labels == lab_tr[i], 0], density=True)
                ax.plot(bins, stats.norm.pdf(bins, np.mean(data[Labels == lab_tr[i], 0]), np.std(data[Labels == lab_tr[i], 0])), linewidth=1.5, color=colors[i, :])
                ax.fill_between(bins, y1=stats.norm.pdf(bins, np.mean(data[Labels == lab_tr[i], 0]), np.std(data[Labels == lab_tr[i], 0])), y2=0, alpha=0.55)

                ax2.plot(data[Labels == lab_tr[i], 0], data[Labels == lab_tr[i], 1], '.', markersize=12, color=colors[i, :], label=lab_tr[i])

                _, bins = np.histogram(data[Labels == lab_tr[i], 1], density=True)
                ax3.plot(stats.norm.pdf(bins, np.mean(data[Labels == lab_tr[i], 1]), np.std(data[Labels == lab_tr[i], 1])), bins, linewidth=1.5, color=colors[i, :])
                ax3.fill_betweenx(bins, stats.norm.pdf(bins, np.mean(data[Labels == lab_tr[i], 1]), np.std(data[Labels == lab_tr[i], 1])), 0, alpha=0.6, color=colors[i, :])
            ax2.legend(title='Class', ncol=3, handlelength=0.8, handletextpad=0.2), ax2.set(xlabel='Feature 1', ylabel='Feature 2')

        elif data.shape[1] > 2:
            ax = fig.add_axes((0.29, 0.78, 0.45, 0.16))
            ax1 = fig.add_axes((0.07, 0.2, 0.12, 0.47))
            ax2 = fig.add_axes((0.24, 0.1, 0.55, 0.65), projection="3d")
            ax3 = fig.add_axes((0.806, 0.2, 0.11, 0.47))

            for i in range(0, len(lab_tr)):
                _, bins = np.histogram(data[Labels == lab_tr[i], 0], density=True)
                ax.plot(bins, stats.norm.pdf(bins, np.mean(data[Labels == lab_tr[i], 0]), np.std(data[Labels == lab_tr[i], 0])), linewidth=2.5, color=colors[i, :])
                ax.fill_between(bins, y1=stats.norm.pdf(bins, np.mean(data[Labels == lab_tr[i], 0]), np.std(data[Labels == lab_tr[i], 0])), y2=0, alpha=0.6, color=colors[i, :])

                _, bins = np.histogram(data[Labels == lab_tr[i], 1], density=True)
                ax1.plot(-stats.norm.pdf(bins, np.mean(data[Labels == lab_tr[i], 1]), np.std(data[Labels == lab_tr[i], 1])), bins, linewidth=2.5, color=colors[i, :])
                ax1.fill_betweenx(bins, 0, -stats.norm.pdf(bins, np.mean(data[Labels == lab_tr[i], 1]), np.std(data[Labels == lab_tr[i], 1])), alpha=0.6, color=colors[i, :])

                ax2.plot3D(data[Labels == lab_tr[i], 0], data[Labels == lab_tr[i], 1], data[Labels == lab_tr[i], 2], '.', markersize=12, color=colors[i, :], label=lab_tr[i])

                _, bins = np.histogram(data[Labels == lab_tr[i], 2], density=True)
                ax3.plot(stats.norm.pdf(bins, np.mean(data[Labels == lab_tr[i], 2]), np.std(data[Labels == lab_tr[i], 2])), bins, linewidth=2.5, color=colors[i, :])
                ax3.fill_betweenx(bins, stats.norm.pdf(bins, np.mean(data[Labels == lab_tr[i], 2]), np.std(data[Labels == lab_tr[i], 2])), 0, alpha=0.6, color=colors[i, :])

            ax2.view_init(10, -100), ax.set_zorder(1), ax2.margins(x=0), ax2.margins(y=0), ax2.margins(z=0),
            ax1.tick_params(bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False), ax1.spines[['top', 'left', 'bottom']].set_visible(False)
            ax2.yaxis.set_ticks(np.linspace(ax2.get_yticks()[1], ax2.get_yticks()[-2], int(len(ax2.get_yticks()) / 2), dtype='int'))
            fig.subplots_adjust(top=1, bottom=0, left=0, right=1, wspace=0, hspace=0),
            ax2.dist = 8.2
            ax2.set(xlabel='Feature 1', ylabel='Feature 2', zlabel='Feature 3')
            pos1 = ax.get_position()
            pos2 = ax2.get_position()
            ax.set_position([0.29, np.abs(pos1.y0-pos2.y0), 0.45, 0.16])
            fig.suptitle(title, fontsize=13, y=pos1.y0+0.1)
            ax2.legend(title='Class', loc=7, ncol=3, handlelength=0.8, handletextpad=0.2)  # bbox_to_anchor=(0.1, pos1.x1-0.02, pos1.x1-0.02, 0)
        ax.tick_params(bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False), ax.spines[['top', 'right', 'left']].set_visible(False)
        ax3.tick_params(bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False), ax3.spines[['top', 'right', 'bottom']].set_visible(False)
    plt.show()
