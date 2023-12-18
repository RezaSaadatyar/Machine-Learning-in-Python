import numpy as np
import seaborn as sns
from scipy import stats
from sklearn import preprocessing
from matplotlib import pyplot as plt

def plot_features(data, labels, fig_size=(4, 3), title="Data raw", normalize_active="on"):
   # --------------------------------------------------- Data transpose --------------------------------------------------
   if data.shape[0] < data.shape[1]: 
       data = data.T 
   # ------------------------------------------------- Unique labels -----------------------------------------------------    
   lab = np.unique(labels)
   colors = np.array(sns.color_palette("bright", len(lab)))
   # ------------------------------------------------- Data normalization ------------------------------------------------ 
   if normalize_active == "on":                          
        norm = preprocessing.MinMaxScaler()
        data = norm.fit_transform(data)
   # ----------------------------------------------------- Plot ---------------------------------------------------------- 
   if data.shape[1] < 3:
       fig = plt.figure(figsize=fig_size)
       
   if data.shape[1] == 1:
      
      grid = plt.GridSpec(4, 4, hspace=0.06, wspace=0.06)
      ax = fig.add_subplot(grid[1:, :3])
      ax1 = fig.add_subplot(grid[0, :3], yticklabels=[], sharex=ax)
      
      for i in range(0, len(lab)):
         
         tim = np.linspace(np.min(data[labels==lab[i], 0]), np.max(data[labels==lab[i], 0]), num=len(data[labels==lab[i], 0]), retstep=True)
         ax.plot(tim[0], data[labels==lab[i], 0], '.', markersize=10, color=colors[i, :], label=f"{lab[i]+1}")
         
         _, bins = np.histogram(data[labels==lab[i], 0], density=True)
         ax1.plot(bins, stats.norm.pdf(bins, np.mean(data[labels==lab[i], 0]), np.std(data[labels==lab[i], 0])), linewidth=1.5, color=colors[i, :])
         ax1.fill_between(bins, y1=stats.norm.pdf(bins, np.mean(data[labels==lab[i], 0]), np.std(data[labels==lab[i], 0])), y2=0, alpha=0.4)
         
      ax.set_xlabel('Feature 1', fontsize=10, va='center')
      ax.tick_params(axis='x', length=1.5, width=1, which='both', bottom=True, top=False, labelbottom=True, labeltop=False, pad=0)
      ax.tick_params(axis='y', length=1.5, width=1, which="both", bottom=False, top=False, labelbottom=True, labeltop=True, pad=0)
     
   elif data.shape[1] < 3:
       
       grid = plt.GridSpec(4, 4, hspace=0.06, wspace=0.06)
       ax = fig.add_subplot(grid[1:, :3])
       ax1 = fig.add_subplot(grid[0, :3], yticklabels=[], sharex=ax)
       ax2 = fig.add_subplot(grid[1:, 3], xticklabels=[], sharey=ax)
       
       for i in range(0, len(lab)):
           
           ax.plot(data[labels==lab[i], 0], data[labels==lab[i], 1], '.', markersize=10, color=colors[i, :], label=f"{lab[i]+1}")
           
           _, bins = np.histogram(data[labels==lab[i], 0], density=True)
           ax1.plot(bins, stats.norm.pdf(bins, np.mean(data[labels==lab[i], 0]), np.std(data[labels==lab[i], 0])), linewidth=1.5, color=colors[i, :])
           ax1.fill_between(bins, y1=stats.norm.pdf(bins, np.mean(data[labels==lab[i], 0]), np.std(data[labels==lab[i], 0])), y2=0, alpha=0.4)
           
           _, bins = np.histogram(data[labels==lab[i], 1], density=True)
           ax2.plot(stats.norm.pdf(bins, np.mean(data[labels==lab[i], 1]), np.std(data[labels==lab[i], 1])), bins, linewidth=2.5, color=colors[i, :])
           ax2.fill_betweenx(bins, stats.norm.pdf(bins, np.mean(data[labels==lab[i], 1]), np.std(data[labels==lab[i], 1])), 0, alpha=0.4, color=colors[i, :])
        
       ax2.spines[['top', 'right', 'bottom']].set_visible(False), 
       ax2.tick_params(bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
       ax.set_xlabel('Feature 1', fontsize=10, va='center'), ax.set_ylabel('Feature 2', fontsize=10, rotation=90, va='center')
       ax.tick_params(axis='x', length=1.5, width=1, which='both', bottom=True, top=False, labelbottom=True, labeltop=False, pad=0)
       ax.tick_params(axis='y', length=1.5, width=1, which="both", bottom=False, top=False, labelbottom=True, labeltop=True, pad=0)
           
   elif data.shape[1] > 2:
      fig = plt.figure(figsize=[4, 3])
      ax = fig.add_axes((0.02, -0.05, 0.9, 0.9), projection="3d")
      ax1 = fig.add_axes((0.22, 0.67, 0.52, 0.16))
      ax2 = fig.add_axes((0.8, 0.18, 0.13, 0.47))
      ax3 = fig.add_axes((-0.05, 0.18, 0.13, 0.47))
      
      for i in range(0, len(lab)):
         
         ax.plot3D(data[labels==lab[i], 0], data[labels==lab[i], 1], data[labels==lab[i], 2], '.', markersize=10, color=colors[i, :], label=f"{lab[i]+1}")
         
         _, bins = np.histogram(data[labels==lab[i], 0], density=True)
         ax1.plot(bins, stats.norm.pdf(bins, np.mean(data[labels==lab[i], 0]), np.std(data[labels==lab[i], 0])), linewidth=2.5, color=colors[i, :])
         ax1.fill_between(bins, y1=stats.norm.pdf(bins, np.mean(data[labels==lab[i], 0]), np.std(data[labels==lab[i], 0])), y2=0, alpha=0.4, color=colors[i, :])
         
         _, bins = np.histogram(data[labels==lab[i], 1], density=True)
         ax2.plot(stats.norm.pdf(bins, np.mean(data[labels==lab[i], 1]), np.std(data[labels==lab[i], 1])), bins, linewidth=2.5, color=colors[i, :])
         ax2.fill_betweenx(bins, stats.norm.pdf(bins, np.mean(data[labels==lab[i], 1]), np.std(data[labels==lab[i], 1])), 0, alpha=0.4, color=colors[i, :])
         
         _, bins = np.histogram(data[labels==lab[i], 2], density=True)
         ax3.plot(-stats.norm.pdf(bins, np.mean(data[labels==lab[i], 2]), np.std(data[labels==lab[i], 2])), bins, linewidth=2.5, color=colors[i, :])
         ax3.fill_betweenx(bins, 0, -stats.norm.pdf(bins, np.mean(data[labels==lab[i], 2]), np.std(data[labels==lab[i], 2])), alpha=0.4, color=colors[i, :])
      
      ax.view_init(5, -120)
      ax.set_xlabel('Feature 1', fontsize=10, va='center'), ax.set_ylabel('Feature 2', fontsize=10, rotation=90, va='center')
      ax.set_zlabel('Feature 3', fontsize=10, labelpad=-5, va='center') # labelpad=1,
      ax.tick_params(axis='x', length=1.5, width=1, which='both', bottom=True, top=False, labelbottom=True, labeltop=False, pad=-6, rotation=-90)
      ax.tick_params(axis='y', length=1.5, width=1, which="both", bottom=False, top=False, labelbottom=True, labeltop=True, pad=-6, rotation=90)
      ax.tick_params(axis='z', which='both', bottom=False, top=False, labelbottom=True, labeltop=False, pad=-2)
      
      ax2.spines[['top', 'right', 'bottom']].set_visible(False), ax3.spines[['top', 'bottom', 'left']].set_visible(False)
      ax2.tick_params(bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
      ax3.tick_params(bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)

   ax.grid(True, linestyle='--', which='major', color='grey', alpha=0.3)
   ax1.set_title(title, fontsize=10, pad=0, y=1)
   ax1.spines[['top', 'left', 'right']].set_visible(False),    
   ax1.tick_params(bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
   ax.legend(title='Class', fontsize=9, loc=5, ncol=3, handlelength=0, handletextpad=0.25, frameon=True, labelcolor='linecolor') 

   fig.subplots_adjust(wspace=0, hspace=0), plt.autoscale(enable=True, axis="x",tight=True)
   # ax.yaxis.set_ticks(np.linspace(ax.get_yticks()[1], ax.get_yticks()[-2], int(len(ax.get_yticks()) / 2), dtype='int'))
   # ax.tick_params(direction='in', colors='grey', grid_color='r', grid_alpha=0.5)
