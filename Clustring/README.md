**Clustring**<br/>
Clustering is a type of unsupervised machine learning technique used to group similar data points together based on certain criteria or features. Clustering methods aim to discover hidden patterns or structures within the data. 

**clustering methods:**
 - `K-Means Clustering:`
   - It partitions the data into K clusters, where K is a user-defined parameter.
   - It minimizes the sum of squared distances between data points and the centroid of their assigned cluster.
   - K-Means is efficient and works well when clusters are spherical and have roughly equal sizes.
- `Hierarchical Clustering:`
   - Hierarchical clustering creates a tree-like structure of nested clusters, known as a dendrogram.
   - It does not require the user to specify the number of clusters beforehand.
   - Agglomerative hierarchical clustering starts with individual data points as clusters and merges them into larger clusters, while divisive hierarchical clustering starts with a single cluster and recursively divides it into smaller clusters.
- `DBSCAN (Density-Based Spatial Clustering of Applications with Noise):`
   - DBSCAN groups data points that are close together and separates data points in less dense regions.
   - It automatically determines the number of clusters based on density.
   - DBSCAN is robust to noise and can find clusters of arbitrary shapes.

4. Gaussian Mixture Model (GMM):
   - GMM assumes that the data is generated from a mixture of multiple Gaussian distributions.
   - It estimates the parameters of these Gaussian distributions to model the data.
   - GMM can be used for density estimation as well as clustering.

5. Mean Shift Clustering:
   - Mean Shift is a non-parametric clustering algorithm that finds modes or peaks in the data distribution.
   - It iteratively shifts data points towards the mode of their local density estimate.
   - It is effective in finding clusters with irregular shapes.

6. Spectral Clustering:
   - Spectral clustering transforms the data into a lower-dimensional space using techniques from linear algebra (e.g., eigenvalue decomposition) and then applies K-Means or another clustering algorithm.
   - It can discover non-convex clusters and is suitable for data with complex structures.

7. Density Peak Clustering (DPC):
   - DPC identifies cluster centers and assigns data points to clusters based on the density of data points around these centers.
   - It is particularly useful for datasets with varying cluster densities.

8. Affinity Propagation:
   - Affinity Propagation is a message-passing algorithm that identifies data points that are exemplars of clusters.
   - It doesn't require the specification of the number of clusters and can discover unevenly sized clusters.

9. Self-Organizing Maps (SOM):
   - SOM is a neural network-based clustering technique that maps high-dimensional data to a lower-dimensional grid.
   - It preserves the topological relationships between data points and can be used for visualization.

Each clustering method has its own advantages and limitations, and the choice of method depends on the nature of the data and the specific problem you are trying to solve. It's often a good idea to experiment with multiple clustering algorithms to see which one performs best for your dataset.