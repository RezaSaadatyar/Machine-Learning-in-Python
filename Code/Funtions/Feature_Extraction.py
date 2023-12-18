import numpy as np
from Plot_Features import plot_features
from sklearn import decomposition, discriminant_analysis, manifold


# ================================================ Feature extraction ===================================================== 
def feature_extraction(data, labels, number_feature, max_iter, number_neighbors_Isomap, perplexity_TSNE, n_iter_TSNE, learning_rate_TSNE,type_feature_extraction, kernel_PCA, solver_LDA, display_figure="off"):
        
        if type_feature_extraction == 'PCA':                            # Principal component analysis
                mod = decomposition.KernelPCA(n_components=number_feature, kernel=kernel_PCA)
                output_feature = mod.fit_transform(data)
        elif type_feature_extraction == 'LDA':                          # Linear discriminant analysis
                mod = discriminant_analysis.LinearDiscriminantAnalysis(n_components=len(np.unique(labels)) - 1, solver=solver_LDA)
                output_feature = mod.fit_transform(data, labels)
                # mod.explained_variance_ratio_
        elif type_feature_extraction == 'ICA':                          # Independent component analysis
                mod = decomposition.FastICA(n_components=number_feature, max_iter=max_iter)
                output_feature = mod.fit_transform(data)
        elif type_feature_extraction == 'SVD':                          # Singular value decomposition
                mod = decomposition.TruncatedSVD(n_components=number_feature)
                output_feature = mod.fit_transform(data)
        elif type_feature_extraction == 'TSNE':                         # T-distributed Stochastic Neighbor Embedding (T-SNE)
                mod = manifold.TSNE(n_components=number_feature, perplexity=perplexity_TSNE, learning_rate=learning_rate_TSNE, verbose=0, n_iter=n_iter_TSNE, init='pca', random_state=0)
                output_feature = mod.fit_transform(data)
        elif type_feature_extraction == 'FA':                           # Factor Analysis
                mod = decomposition.FactorAnalysis(n_components=number_feature, max_iter=max_iter)
                output_feature = mod.fit_transform(data)
        elif type_feature_extraction == 'Isomap':                       # Isometric Feature Mapping
                mod = manifold.Isomap(n_neighbors=number_neighbors_Isomap, n_components=number_feature)
                output_feature = mod.fit_transform(data)
        
        if display_figure == "on":
                plot_features(output_feature, labels, fig_size=(4, 3), title=type_feature_extraction, normalize_active="on")
        
        return output_feature