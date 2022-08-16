####   Machine Learning in Python
>***This repository Covers:***
>- 1. A brief about types of classification & clustering algorithms
>- 2. Preparing the data
>- 3. Training the model
>- 4. Prediction and performance check
>- 5. Iris dataset classification & clustering example<br/>

**Standardization & Normalization:**<br/>Scaling is required when we use any machine learning algorithm that require *gradient calculation*. Examples of machine learning algorithms that require gradient calculation are: *linear/logistic regression* and *artificial neural networks*. Scaling  is not required for distance-based and tree-based algorithms such as *K-means clustering, Support Vector Machines, K Nearest Neighbors, decision tree, random forest* and *XG-Boost*. Having different sales for each feature will result in a different step size which in turn jeopardizes the proess of reaching a minimum point.

A machine learning algorithm (such as **classification, clustering or regression**) uses a training dataset to determine weight factors that can be applied to unseen data for predictive purposes. Before implementing a ML algorithm, it is necessary to select only relevant features in the training dataset. The process of transforming a dataset in order to select only relevant features necessary for training is called **dimensionality reduction**.


> **Dimensionality Reduction:**<br/>Feature seletion and dimensionality reduction are important because of three main reasons:
>  1. Prevents overfitting: A high-dimensional dataset having too many features can sometimes lead to overfitting (model captures both real and random effets).
>  2. Simplicity: An over-complex model having too many features can be hard to interpret especially when features are correlated with each other.
>  3. Computational efficiency: A model trained on a lower-dimensional dataset is omputationally efficient (execution of algorithm reuires less computational time.<br/> 
>
>**Feature extraction:**<br/>
>   - Principal Component Analysis (PCA) 
>   - Linear Discrimination Analysis (LDA)
>   - Independent component analysis (ICA)
>   - Singular value decomposition (SVD)
>   - Factor analysis (FA) 
>   - Isometric Feature Mapping (Isomap)
>   - T-distributed stochastic neighbor embedding (TDSNE)
>   
> **Feature selection methods:**<br/>
> - ***Filter methods:***
>      - Univariate
>        - Variance threshold (VT)
>        - Mutual information (MI)
>        - Chi-square test (Chi-square)
>        -  fisher_score (FS)
>     - Multi-variate
>        - Pearson correlation
>  - ***Warpper method:*** 
>       - Forward feature selection (FFS)
>       - Exhaustive Feature Selection (EFS)
>       - Recursive feature elimination (RFE)
>       - Backward feature selection (BFS) 
>   - ***Embedded method:*** 
>     - Random forest (RF)
>     - Tree-based feature selection (TFS)
>     - L1- regularized logistic regression 
>     
>**Further information:**<br/> 
> ***PCA:*** It is a dimension-reduction tool that can be used to reduce a large set of variables to a small set that still contains most of the information in the largest set (Maximising the variance of the whole set).<br/>
> ***LDA:*** Maximising the distance between groups.<br/>
> ***TDSNE:*** It is a tool to visualize a high dimensional data. It converts similarities between data points to point probabilities and tries to minimize the *Kullback-Leibler (KL)* divergence between the joint probabilities of the low dimensional embedding and high dimensional data.<br/> 
> ***Univariate:*** The univariate filter methods are the type of methods where individual features are ranked according to specific criteria.The top N features are then selected.<br/>
> ***VT:***<br/>1. Compute the variance of feature<br/> 2. Assume that features with a higher variance may contain more useful information<br/> 3. Fast method but does not take the relationship among features into account.<br/>
> ***Multi-variate:*** Multivariate filter methods are capable of removing redundant feature form the data since they take the mutual relationship between the features into account.<br/>
> ***BFS:***<br/> 1. Choose a significances level (e.g., SL = 0.05 with a 95% confidence)<br/>2. Fit a full model including all the features<br/> 3. Consider the features with the highrst p-value. if the p-value > significance level the go to step 4, ptherwise terminate the process.<br/> 4. Remove the feature which is under consideration.<br/> 5. Fit a model without this feature. Repeat the entire process from step 3.  
> ***The embedded method*** solves both issues we encountered with the filterand wrapper methods by combining their advantages.<br/>They take into consideration the interaction of features like wrapper methods do.<br/>They are faster like filter methods.<br/>They are more accurate than methods.<br/>They find the feature subset for the algorithm being trained.<br/>They are much less prone to overfitting.





>| Filter method | Wrapper method | Embedded method|
>| --------------- | -------------- |----------------|
>|Uses proxy measure| Uses predictive model |Feature selection is embedded in the model building phase|
>|Computationally faster |Slower |Medium  |
>|Avoids overfitting |Prone to overfitting|Less Prone to overfitting|
>|Sometimes may fail to select best features |Better| Good performance|




>**Types of Machine Learning:**
>  - Supervised Learning
>    - Classification
>       - AdaBoost
>       - Naive Bayes
>       - Random Forest
>       - Decision Trees
>       - K-Nearest Neighbors
>       - Logistic Regression
>       - Multilayer perceptron
>       - Support Vector Machins
>       - Radial Basis Function (RBF) 
>       - Linear Discriminant Analysis (LDA)
>    - Regression
>       - Lasso Regression
>       - Ridge Regression
>       - Linear Regression
>       - Decision Trees Regression
>       - Neural Network Regression
>       - Support Vector Regression
>  - Unsupervised Learning
>     - Clustering
>       - DBSCAN
>       - K-Means
>       - Mean-shift
>       - Fuzzy c-means
>       - Gaussian Mixture
>       - Agglomerative Hierarchial
>       - Principal Component Analysis
>  - Reinforcement Learning
>     - Decision Making
>       - Q-Learning
>       - R Learning
>       - TD Learning

>**Types of Classification Algorithms:**<br/>
> - Logistic Regression:<br/>This method is most useful when examining the influence of several independent variables on a single outcome.Only works when the predicted variable is binary, assumes independent predictors, and assumes no missing values
> - Linear Discriminant Analysis<br/> LDA is a linear model for classification and dimensionality reduction.  Most commonly used for feature extraction in pattern classification problems. In LDA, data is projected to a space where the variance between classes is maximized, but the variance within classes is minimized.
>    - ***LDA VS PCA:***  
>        - LDA classifies data vs PCA classifies features
>        - LDa is a supervised learning technique vs PCA is a unsupervised learning techinque
>        - LDA projects the data in a direction which proides maximum inter-class seperability vs PCA projects the data in a direction of maximum variation
>        - LDA can reduce data up to 'number of class-1' dimension vs PCA can be used to reduce data up any dimentions 
>-  Decision Trees<br/>The decision tree can create complex trees that do not generalize well, and it can become unstable if a small variation in the data changes it completely.
>-  Random Forest<br/>  It reduces over-fitting and is more accurate than decision trees in most cases. It has a slow real-time prediction, is difficult to implement, and has a complex algorithm.
>-  Naive Bayes:<br/>
>   This algorithm requires a small amount of training data to estimate the necessary parameters. The Naive Bayes classifier is extremely fast compared with more sophisticated methods. In general, Naive Bayes is not a good estimator.
> $$P(Class_j | x) = {P(x | Class_j)*P(Class_j)\over P(x)}$$      $$P(x | Class_j) = P(X_1 | Class_j) * P(X_2 | Class_j) * ... * P(X_k | Class_j)$$    
>- K-Nearest Neighbor<br/>
>  Based on the k nearest neighbours of each point, classification is calculated. In addition to being simple to implement, this algorithm is robust to noisy training data and effective with large training data sets. As it needs to compute the distance between each instance and all the training samples, the computation cost is high.<br/>***KNN(D, d, k):***<br/>1. compute the distance between d and every example in D <br/>2. Choose the k example in D that are nearest to d<br/>3. Assign d the class that is the most frequent class in the majority class<br/>4. Where k is very small, the model is complex and hence we overfit<br/>5. Where k is very large, the model is simple and we underfit
>- Support Vector Machine<br/> Adaptive to high-dimensional spaces and uses a subset of training points for the decision function, making it memory-efficient too.<br/>
>  ***The advantages of SVM are:***
>    - Effective in high dimensional spaces
>    - Still effective in cases where number of dimensions is greater than the nunmber of samples.
>    - Uses a subset of training points in the decision function (called support vectors), so it is also memroy efficient.
>    - Versatile different Kernel functions can be specified for the decision funtion
>- Radial Basis Function (RBF):<br/> RBF networks are similar to two-layer networks. There is a hidden layer that is completely connected to an input. Then, we take the output of the hidden layer perform a weighted sum to get our output.
>- AdaBoost<br/> 
>  It is an ensemble learning method created to improve binary classifier efficiency. AdaBoost uses an iterative approach to learn from the mistakes of weak classifiers, and turn them into strong ones
>     - Boosting algorithm<br/> It is a process that uses a set of machine learning algorithms to combine weak learner to form strong learners in order to increase the accuracy of the model.
>      - 1. The base algorithm reads the data and assigns equal weight to each sample observation.
>      - 2. False predictions are assigned to the next base learner with a higher weightage on these incorrect predictions.
>      - 3. Repeat step 2 until algorithm can correctly classify the output.
>    - Gradient Boosting method<br/>In Gradient Boosting, base learner are generated sequentially in such a way that the present base learner is always more effective than the previous one.
>       - ***XGboost*** is an advanced version of Gradient Boosting method that is designed to focus on computational speed and model efficiency.
 
> **Types of Clustering Algorithms:**<br/>Typically, In unsupervised algorithms, input vectors are used to make inferences without referring to labelled outcomes.
>   - K-means: <br/>During data mining, the K-means algorithm starts with a set of randomly selected centroids, which serve as the starting points for every cluster, and then performs iterative (repetitive) calculations to optimize their positions.
>   - Fuzzy c-means
>     - Based on the cluster center membership, each point is assigned a percentage from 0 to 100 percent. Comparatively, this can be quite powerful compared to traditional hard-threshold clustering, where each point is assigned an exact, crisp label.
 
> **Confusion matrix:**
>   - 1. TP (True Positive): The number of correct classification of positive examples
>   - 2. TN (True Negative): The number of correct classification of negative examples
>   - 3. FP (False Positive): The number of incorrect classification of negative examples
>   - 4. FN (False Negative): The number of incorrect classification of positive examples


> **Clustering:**<br/>We have a set of unlabeled data point x and we intend to find groups of similar objects (based on observed features)<br/>1. High intra-cluster similarity: cohesive within clusters<br/>2. low intra-cluster similarity: distinctive between clusters<br/>
> ***The general approach of clustering algorithms:*** 
>  - 1. Partitional clustering
>       - K-Means algorithm: It is not suitable for disovering clusters that are not hyper-spheres.
>  - 2. Hierarachical clustering 
>       - Agglomerative method
>       - Divisible method
>  - 3. Density-based clustering
>        - Density-Based Spatial Clustering of Applications with Noise (DBSCAN)

>**Types of outliers:**
> - 1. Global outliers
> - 2. Collective outliers
> - 3. Contextual outliers
> 
>**Outlier detection methods:**
> - 1. Supervised methods
> - 2. Unsupervised methods
>      - Clustering-Based
>      - Proximity-Based
>        - Distane based
>        - Density Based
>      - Classification-Based
>      - Statistical
>        - Boxplot method
>        - Histogram method
>        
> ***Install the required packages (if required)***   
> - pip install numpy
> - pip install scipy
> - pip install pandas
> - pip install seaborn
> - pip install matplotlib   
> - pip install scikit-learn

> **There are 6 steps to effective data classification:**
> In short, the steps involved in
![Website Flowchart Template (1)](https://user-images.githubusercontent.com/96347878/184292921-53b07af7-3238-42ff-9c48-226cb2f965ce.png)
> **Step 1:**
> ```
> from sklearn import datasets
> iris = datasets.load_iris()
> Data = iris.data
> Labels = iris.target
> or  # Data, Labels = datasets.make_blobs(150, 4, centers=4, random_state=0)   # The make_blobs() function can be used to generate blobs of points with a Gaussian distribution.
> # Labels = preprocessing.LabelEncoder()   # Encode target labels with value between 0 and n_classes-1 (if required).
> # Labels = Labels.fit_transform(Labels)
> ```
> **Step 2:** <br/>In this section, Iris datasets are used, so filtering and normalization are not necessary.<br/>
> ```
> # ===================================== Step 2: Filtering & Data scaling =============================================
> # Data = filtering(Data, F_low=5, F_high=10, Order=3, Fs=50, btype='bandpass')      # btype:'low', 'high', 'bandpass', 'bandstop'
> # Data = normalize_data(Data, Type_Normalize='MinMaxScaler', Display_Figure='on')   # Type_Normalize:'MinMaxScaler', 'normalize'
> ```
> **Step 3:**<br/>There are two ways to reduce dimensionality:
> - By only keeping the most relevant variables from the original dataset (this technique is called feature selection)
> - Using a smaller set of new variables containing basically the same information as the input variables, each being a combination of the input variables (this technique is called dimensionality reduction).
 ```
 # Data = feature_extraction(Data, Labels, Number_Feature_PCA=3, Type_feature='ICA')     # Feature Extraction  ype_feature=LDA, PCA, ICA
 Data = featureselection(Data, Labels, Number_Feature=3, C_L1FS=0.01, N_estimators_TFS=100,  Type_feature='RFECV')
 ```
 <img width="6687" alt="pca1" src="https://user-images.githubusercontent.com/96347878/184386205-c1f85c85-c327-4cf5-98b8-d6c860598379.png">
