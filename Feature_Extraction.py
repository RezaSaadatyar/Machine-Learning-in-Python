import numpy as np
from Plot_Feature_Extraction_Selection import plot_feature_extraction_selection
from sklearn import decomposition, discriminant_analysis, manifold


def feature_extraction(input_data, labels, number_feature, number_neighbors, type_feature):
    lab_tr = np.unique(labels)

    if type_feature == 'PCA':                            # Principal component analysis
        mod = decomposition.KernelPCA(n_components=number_feature, kernel='linear')
        output_feature = mod.fit_transform(input_data)
    elif type_feature == 'LDA':                          # Linear discriminant analysis
        mod = discriminant_analysis.LinearDiscriminantAnalysis(n_components=len(np.unique(labels)) - 1)
        output_feature = mod.fit_transform(input_data, labels)
        # mod.explained_variance_ratio_
    elif type_feature == 'ICA':                          # Independent component analysis
        mod = decomposition.FastICA(n_components=number_feature)
        output_feature = mod.fit_transform(input_data)
    elif type_feature == 'SVD':                          # Singular value decomposition
        mod = decomposition.TruncatedSVD(n_components=number_feature)
        output_feature = mod.fit_transform(input_data)
    elif type_feature == 'TSNE':                         # T-distributed Stochastic Neighbor Embedding (T-SNE)
        mod = manifold.TSNE(n_components=number_feature, perplexity=15, learning_rate='auto', verbose=2, n_iter=1000, init='pca', random_state=0)
        output_feature = mod.fit_transform(input_data)
    elif type_feature == 'FA':                           # Factor Analysis
        mod = decomposition.FactorAnalysis(n_components=number_feature)
        output_feature = mod.fit_transform(input_data)
    elif type_feature == 'Isomap':                       # Isometric Feature Mapping
        mod = manifold.Isomap(n_neighbors=number_neighbors, n_components=number_feature)
        output_feature = mod.fit_transform(input_data)
    plot_feature_extraction_selection(input_data, output_feature, labels, type_feature)
    return output_feature
