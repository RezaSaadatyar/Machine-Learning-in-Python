from Plot_clustering import plot_cluster
from sklearn import cluster, mixture


def clustering(data, n_clusters, max_iter,type_cluster, thr_brich,  branchfactor_brich, n_neighbor_SpecCluster, minsamples_optics,
               max_dist_optics, batch_size_MBKmeans):
    if type_cluster == 'kmeans':
        mod = cluster.KMeans(n_clusters=n_clusters, max_iter=max_iter, random_state=0)
        labels = mod.fit_predict(data)
    elif type_cluster == 'Agglomerative':
        mod = cluster.AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward')  # affinity: euclidean, manhattan; linkage: ward, single, average, complete
        labels = mod.fit_predict(data)
    elif type_cluster == 'DBSCAN':
        mod = cluster.DBSCAN(eps=1, min_samples=10)
        labels = mod.fit_predict(data)
    elif type_cluster == 'GMM':
        mod = mixture.GaussianMixture(n_components=n_clusters)
        labels = mod.fit_predict(data)
    elif type_cluster == 'Meanshift':
        mod = cluster.MeanShift(max_iter=max_iter)
        labels = mod.fit_predict(data)
        mod.cluster_centers_
    elif type_cluster == 'Birch':
        mod = cluster.Birch(threshold=thr_brich, n_clusters=n_clusters, branching_factor=branchfactor_brich)
        labels = mod.fit_predict(data)
    elif type_cluster == 'SpectralClustering':
        mod = cluster.SpectralClustering(n_clusters=n_clusters, n_neighbors=n_neighbor_SpecCluster)
        labels = mod.fit_predict(data)
    elif type_cluster == 'OPTICS':
        mod = cluster.OPTICS(min_samples=minsamples_optics, max_eps=max_dist_optics, metric='minkowski',)
        labels = mod.fit_predict(data)
    elif type_cluster == 'MiniBatchKMeans':
        mod = cluster.MiniBatchKMeans(n_clusters=n_clusters, max_iter=max_iter, batch_size=batch_size_MBKmeans)
        labels = mod.fit_predict(data)
        mod.cluster_centers_

    plot_cluster(data, labels, type_cluster)

