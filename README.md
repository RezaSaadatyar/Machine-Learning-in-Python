#### Machine Learning in Python
> All classifiers in machine learning - step by step
>- Decision Tree
>- Random Forest
>- Naive Bayes
> $$P(Class_j | x) = {P(x | Class_j)*P(Class_j)\over P(x)}$$      $$P(x | Class_j) = P(X_1 | Class_j) * P(X_2 | Class_j) * ... * P(X_k | Class_j)$$    
>- Gradient Boosting
>- K-Nearest Neighbor (KNN(D, d, k)):
>   1. compute the distance between d and every example in D
>   2. Choose the k example in D that are nearest to d
>   3. Assign d the class that is the most frequent class in the majority class
>   4. Where k is very small, the model is complex and hence we overfit
>   5. Where k is very large, the model is simple and we underfit
>- Logistic Regression
>- Support Vector Machine (SVM)
>- AdaBoost


> **Dimensionality Reduction:**
>
> Feature selection methods:
>   - Filter method: 
>     - Correleation
>     - Mutual Information
>     - t-test
>     -  Univariate feature selestion: It works by selecting the best features based on univariate statistical tests. It can be seen as a preprocessing step to an estimator.
>   - Warpper method: use cross-validation
>     - Sequential forward selection (SFS)
>     - Sequential backward selection (SBS)
>     - Plus-L minus-R selection (LRS) 
>   - Embedded method
>     - L1 (LASSO) regularization
>   
> Filter methods use statistical methods for evaluation of a subset of features while warpper methods use cross validation.
> 
> Feature extraction:
>   - Principal Component Analysis (PCA): It is a dimension-reduction tool that can be used to reduce a large set of variables to a small set that still contains most of the information in the largest set (Maximising the variance of the whole set).
>   - Linear Discrimination Analysis (LDA): Maximising the distance between groups 

> **Clustering:**<br/>We have a set of unlabeled data point x and we intend to find groups of similar objects (based on observed features)<br/>1. High intra-cluster similarity: cohesive within clusters<br/>2. low intra-cluster similarity: distinctive between clusters<br/>
> ***The general approach of clustering algorithms:*** 
>  - 1. Partitional clustering
>       - K-Means algorithm: It is not suitable for disovering clusters that are not hyper-spheres.
>  - 2. Hierarachical clustering 
>       - Agglomerative method
>       - Divisible method
>  - 3. Density-based clustering
>       - Density-Based Spatial Clustering of Applications with Noise (DBSCAN)
>    
> 
>   
