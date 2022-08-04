####   Machine Learning in Python
>***This repository Covers:***
>- 1. A brief about types of classification & clustering algorithms
>- 2. Preparing the data
>- 3. Training the model
>- 4. Prediction and performance check
>- 5. Iris dataset classification & clustering example

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
>       - Linear Discriminant Analysis
>    - Regression
>       - Lasso Regression
>       - Ridge Regression
>       - Linear Regression
>       - Decision Trees Regression
>       - Neural Network Regression
>       - Support Vector Regression
>  - Unsupervised Learning
>     - Clustering
>       - K-Means
>       - Mean-shift
>       - DBSCAN
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
>     
>-  Decision Trees<br/>The decision tree can create complex trees that do not generalize well, and it can become unstable if a small variation in the data changes it completely.
>-  Random Forest<br/>  It reduces over-fitting and is more accurate than decision trees in most cases. It has a slow real-time prediction, is difficult to implement, and has a complex algorithm.
>-  Naive Bayes:<br/>
>   This algorithm requires a small amount of training data to estimate the necessary parameters. The Naive Bayes classifier is extremely fast compared with more sophisticated methods. In general, Naive Bayes is not a good estimator.
> $$P(Class_j | x) = {P(x | Class_j)*P(Class_j)\over P(x)}$$      $$P(x | Class_j) = P(X_1 | Class_j) * P(X_2 | Class_j) * ... * P(X_k | Class_j)$$    
>- K-Nearest Neighbor<br/>
>  Based on the k nearest neighbours of each point, classification is calculated. In addition to being simple to implement, this algorithm is robust to noisy training data and effective with large training data sets. As it needs to compute the distance between each instance and all the training samples, the computation cost is high.<br/>***KNN(D, d, k):***<br/>1. compute the distance between d and every example in D <br/>2. Choose the k example in D that are nearest to d<br/>3. Assign d the class that is the most frequent class in the majority class<br/>4. Where k is very small, the model is complex and hence we overfit<br/>5. Where k is very large, the model is simple and we underfit
>- Support Vector Machine<br/> Adaptive to high-dimensional spaces and uses a subset of training points for the decision function, making it memory-efficient too.
>- AdaBoost<br/> 
>  It is an ensemble learning method created to improve binary classifier efficiency. AdaBoost uses an iterative approach to learn from the mistakes of weak classifiers, and turn them into strong ones
>     - Boosting algorithm<br/> It is a process that uses a set of machine learning algorithms to combine weak learner to form strong learners in order to increase the accuracy of the model.
>      - 1. The base algorithm reads the data and assigns equal weight to each sample observation.
>      - 2. False predictions are assigned to the next base learner with a higher weightage on these incorrect predictions.
>      - 3. Repeat step 2 until algorithm can correctly classify the output.
>    - Gradient Boosting method<br/>In Gradient Boosting, base learner are generated sequentially in such a way that the present base learner is always more effective than the previous one.
>       - ***XGboost*** is an advanced version of Gradient Boosting method that is designed to focus on computational speed and model efficiency.
      
> **Confusion matrix:**
>   - 1. TP (True Positive): The number of correct classification of positive examples
>   - 2. TN (True Negative): The number of correct classification of negative examples
>   - 3. FP (False Positive): The number of incorrect classification of negative examples
>   - 4. FN (False Negative): The number of incorrect classification of positive examples

> **Dimensionality Reduction:**
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
> Feature extraction:<br/>
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
> ***Install the required packages (if required)***   
> - pip install numpy
> - pip install pandas
> - pip install scikit-learn
> - pip install matplotlib   
