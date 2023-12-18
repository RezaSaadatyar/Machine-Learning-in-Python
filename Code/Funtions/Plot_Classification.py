import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from matplotlib.colors import ListedColormap

# from Plot_decision_regions import plot_decision_regions


def plot_classification(data_train, label_train, data_test, label_test, model, k_fold, type_class, display_classification="off", display_normalize_classification="on", fig_size_classification=(5, 3)):
    # --------------------------------------------------- Data transpose -------------------------------------------------
    if display_classification == "on":
        if data_train.shape[0] < data_train.shape[1]:
            data_train = data_train.T
    
        if data_test.shape[0] < data_test.shape[1]:
            data_test = data_test.T   
        # ------------------------------------------ Predict the train & test labels -------------------------------------
        miss_class_train = label_train - model.predict(data_train)      # predict the train labels
        miss_class_train = len(miss_class_train[miss_class_train != 0])
        
        miss_class_test = label_test - model.predict(data_test)         # predict the test labels
        miss_class_test = len(miss_class_test[miss_class_test != 0]) 
        lab = np.unique(label_train)
        # ------------------------------------------------- Data normalization -------------------------------------------
        if display_normalize_classification=="on":
            if np.max(data_train) > 1:
                norm = preprocessing.MinMaxScaler()
                data_train = norm.fit_transform(data_train)
                data_test = norm.fit_transform(data_test)
        # --------------------------------------------------- Plot -------------------------------------------------------
        if data_train.shape[1] < 3:
   
            resolution = 0.03
            x_combined = np.vstack((data_train, data_test))
            y_combined = np.hstack((label_train, label_test))
            x1_min, x1_max = x_combined[:, 0].min() - 1, x_combined[:, 0].max() + 1
            x2_min, x2_max = x_combined[:, 1].min() - 1, x_combined[:, 1].max() + 1
            xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
            z = model.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
            
            fig, axs = plt.subplots(1, 2, sharey='row', figsize=fig_size_classification, constrained_layout=True)
            colors = list(reversed(sns.color_palette("bright", len(lab)).as_hex()))
            cmp = ListedColormap(colors[:len(lab)])
            
            axs[0].contourf(xx1, xx2, z.reshape(xx1.shape), alpha=0.2, cmap=cmp)
            axs[0].set_xlim(xx1.min(), xx1.max())
            axs[0].set_ylim(xx2.min(), xx2.max())

            axs[1].contourf(xx1, xx2, z.reshape(xx1.shape), alpha=0.2, cmap=cmp)
            axs[1].set_xlim(xx1.min(), xx1.max())
            axs[1].set_ylim(xx2.min(), xx2.max())

            for i in range(0, len(np.unique(y_combined))):
                axs[0].plot(data_train[label_train==lab[i], 0], data_train[label_train==lab[i], 1], '.', color=colors[i], label=lab[i], markersize=10)
                axs[1].plot(data_test[label_test==lab[i], 0], data_test[label_test==lab[i], 1], '.', color=colors[i], label=lab[i], markersize=10)
            
            axs[0].tick_params(axis='x', length=1.5, width=1, which='both', bottom=True, top=False, labelbottom=True, labeltop=False, pad=0.5)
            axs[0].tick_params(axis='y', length=1.5, width=1, which="both", bottom=False, top=False, labelbottom=True, labeltop=True, pad=0.5)
            axs[1].tick_params(axis='x', length=1.5, width=1, which='both', bottom=True, top=False, labelbottom=True, labeltop=False, pad=0.5)
            axs[1].tick_params(axis='y', length=1.5, width=1, which="both", bottom=False, top=False, labelbottom=True, labeltop=True, pad=0.5)
            
            axs[0].set_xlabel('Feature 1',  fontsize=10, va='center'), axs[0].set_ylabel('Feature 2', fontsize=10, va='center')
            axs[0].set_title(f"Traininng; Miss_classes: {miss_class_train}#", loc='left', pad=0, fontsize=10) 
            axs[0].legend(title="Class", loc="best", fontsize=9, ncol=3, frameon=True, labelcolor='linecolor', handlelength=0.2, handletextpad=0.2)
            
            axs[1].set_xlabel('Feature 1',  fontsize=10, va='center'), axs[1].set_title(f"Test; Miss_classes: {miss_class_test}#", loc='right', pad=0, fontsize=10)
            
            fig.suptitle(f"{type_class}; {k_fold}_fold", fontsize=11, fontweight='normal', color='black', va='top')
            
        elif data_train.shape[1] > 2:
           
            x_train = data_train[:, 0:3]
            x_test = data_test[:, 0:3]
        
            fig = plt.figure(figsize=(7.5,3.5))
            ax1 = fig.add_subplot(121, projection='3d')
            ax2 = fig.add_subplot(122, projection='3d')

            for i in range(0, len(lab)):
                ax1.plot3D(x_train[label_train==lab[i], 0], x_train[label_train==lab[i], 1], x_train[label_train==lab[i], 2], '.', label=lab[i])
                ax2.plot3D(x_test[label_test==lab[i], 0], x_test[label_test==lab[i], 1], x_test[label_test==lab[i], 2], '.')

            ax1.view_init(5, -120), ax2.view_init(5, -120) 
            ax1.set_xlabel('Feature 1', fontsize=10, va='center'), ax2.set_xlabel('Feature 1', fontsize=10, va='center')
            ax1.set_ylabel('Feature 2', labelpad=1, fontsize=10, va='center'), ax2.set_ylabel('Feature 2', labelpad=1, fontsize=10, va='center')
            ax1.set_zlabel('Feature 3', labelpad=-6, fontsize=10, va='center', rotation=45)  
            
            ax1.tick_params(axis='x', length=1.5, width=1, which='both', bottom=True, top=False, labelbottom=True, labeltop=False, pad=-4)
            ax2.tick_params(axis='x', length=1.5, width=1, which='both', bottom=True, top=False, labelbottom=True, labeltop=False, pad=-4)
            ax1.tick_params(axis='y', length=2, width=1, which="both", bottom=True, top=False, labelbottom=True, labeltop=True, pad=-6, rotation=90)
            ax2.tick_params(axis='y', length=2, width=1, which="both", bottom=True, top=False, labelbottom=True, labeltop=True, pad=-6, rotation=90)
            ax1.tick_params(axis='z', length=2, width=1, which='both', bottom=False, top=False, labelbottom=True, labeltop=False, pad=-2)
            ax2.tick_params(axis='z', length=2, width=1, which='both', bottom=False, top=False, labelbottom=True, labeltop=False, pad=-2)
            
            ax1.legend(title="Class", loc=5, fontsize=9, ncol=3, frameon=True, labelcolor='linecolor', handlelength=0.2, handletextpad=0)
            ax1.set_title(f"Traing; Miss_classes: {miss_class_train }#", loc='right', pad=0, y=0.8, fontsize=10)
            ax2.set_title(f"Test; Miss_classes: {miss_class_test}#", loc='right', pad=0, y=0.8, fontsize=10),
            fig.suptitle(f"{type_class}; {k_fold}_fold", fontsize=11, x=0.51,  y=0.9, fontweight='normal', color='black', va='top')
                
            plt.tight_layout(w_pad=-1, h_pad=0), plt.subplots_adjust(top=1, bottom=0, left=0 ,wspace=-0.1, hspace=0)
            
        plt.autoscale(enable=True, axis="x", tight=True)
        # ax.tick_params(direction='in', length=6, width=2, colors='grey', grid_color='r', grid_alpha=0.5)
    