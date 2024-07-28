import numpy as np
from Funtions import Plot_features
from sklearn import decomposition, discriminant_analysis, manifold

# ======================================== Feature extraction ============================================ 
def feature_extraction(data, labels, number_feature, max_iter=100, number_neighbors_Isomap=10, 
                       perplexity_TSNE=5, n_iter_TSNE=200, learning_rate_TSNE="auto", type_feature_extraction="pca",
                       kernel_PCA="linear", solver_LDA="svd", display_figure="off"):
        
        if type_feature_extraction.lower() == 'pca':         # Principal component analysis
                mod = decomposition.KernelPCA(n_components=number_feature, kernel=kernel_PCA)
                output_feature = mod.fit_transform(data)
        elif type_feature_extraction.lower() == 'lda':       # Linear discriminant analysis
                mod = discriminant_analysis.LinearDiscriminantAnalysis(n_components=len(np.unique(labels))
                                                                       - 1, solver=solver_LDA)
                output_feature = mod.fit_transform(data, labels)
                # mod.explained_variance_ratio_
        elif type_feature_extraction.lower() == 'ica':       # Independent component analysis
                mod = decomposition.FastICA(n_components=number_feature, max_iter=max_iter)
                output_feature = mod.fit_transform(data)
        elif type_feature_extraction.lower() == 'svd':       # Singular value decomposition
                mod = decomposition.TruncatedSVD(n_components=number_feature)
                output_feature = mod.fit_transform(data)
        elif type_feature_extraction.lower() == 'tsne':      # T-distributed Stochastic Neighbor Embedding
                mod = manifold.TSNE(n_components=number_feature, perplexity=perplexity_TSNE, 
                                    learning_rate=learning_rate_TSNE, verbose=0, n_iter=n_iter_TSNE, 
                                    init='pca', random_state=0)
                output_feature = mod.fit_transform(data)
        elif type_feature_extraction.lower() == 'fa':        # Factor Analysis
                mod = decomposition.FactorAnalysis(n_components=number_feature, max_iter=max_iter)
                output_feature = mod.fit_transform(data)
        elif type_feature_extraction.lower() == 'isomap':    # Isometric Feature Mapping
                mod = manifold.Isomap(n_neighbors=number_neighbors_Isomap, n_components=number_feature)
                output_feature = mod.fit_transform(data)
        
        if display_figure == "on":
                Plot_features.plot_features(output_feature, labels, title=type_feature_extraction)
        
        return output_feature
