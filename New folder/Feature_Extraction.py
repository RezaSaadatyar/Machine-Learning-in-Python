import numpy as np
from Plot_Feature_Extraction_Selection import plot_feature_extraction_selection
from sklearn import decomposition, discriminant_analysis


def feature_extraction(input_data, Labels, Number_Feature_PCA, Type_feature='PCA'):
    if input_data.shape[0] < input_data.shape[1]:
        input_data = input_data.T
    lab_tr = np.unique(Labels)

    if Type_feature == 'PCA':
        mod = decomposition.KernelPCA(n_components=Number_Feature_PCA, kernel='linear')
        output_feature = mod.fit_transform(input_data)
    elif Type_feature == 'LDA':
        mod = discriminant_analysis.LinearDiscriminantAnalysis(n_components=len(np.unique(Labels)) - 1)
        output_feature = mod.fit_transform(input_data, Labels)
        mod.explained_variance_ratio_
    elif Type_feature == 'ICA':
        mod = decomposition.FastICA(n_components=Number_Feature_PCA)
        output_feature = mod.fit_transform(input_data)
    plot_feature_extraction_selection(input_data, output_feature, Labels, Type_feature)

    return output_feature
