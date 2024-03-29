**Classification**<br/>

**Binary Classification:**<br/>
It is a type of supervised learning algorithm used in machine learning. It aims to classify new observations into one of two possible outcomes, typically denoted as 0 or 1, true or false, positive or negative, and so on.

**How Binary Classification Works?**<br/>The process of binary classification involves training an algorithm on a dataset that has labeled data points with binary labels. The algorithm then learns how to associate input features with their respective binary labels. Once the algorithm is trained, it can be utilized to predict binary labels for new and unseen data points.

**Common Binary Classification Models:**<br/>
- `Logistic Regression:` This is utilized in problems involving binary classification, where the output variable has two categorical values.
- `Neural Networks:` This algorithm clusters input, recognizes patterns, and interprets sensory data. However, neural networks require significant computational resources, making it difficult to fit them when dealing with thousands of observations.
 - `Support Vector Machines:` Support vector machines are used for classification by creating a hyperplane that maximizes the distance between two classes of data points. This hyperplane, known as the decision boundary, separates the classes of data points on either side. For example, it could separate oranges from apples.
 - `Decision Trees:` Decision trees are interpretive tree-like structures used for binary classification tasks, with each node representing a decision based on a feature, leading to one of two possible outcomes.
- `Random Forest:` It is a versatile form of supervised machine learning that can be applied to both classification and regression tasks. It involves an ensemble of decision trees that work together to enhance accuracy and minimize the risk of overfitting.
- `Naive Bayes:` The Naive Bayes algorithm operates under the assumption that the input variables or features are independent of each other, given the class label. However, this assumption is considered "naive" as it does not account for the possibility of correlations between features in reality. The three primary variants of Naive Bayes are Gaussian Naive Bayes, Multinomial Naive Bayes, and Bernoulli Naive Bayes.
- `K-Nearest Neighbors (KNN):` KNN is a classification algorithm that assigns a class label to a data point based on the majority class among its k-nearest neighbors in feature space.
- `Gradient Boosting Machines (e.g., XGBoost, LightGBM):` Gradient boosting models, such as XGBoost and LightGBM, are ensemble techniques that use decision trees to correct errors made by previous trees. They are extremely effective and commonly used in binary classification competitions.
- `Adaboost:` Adaboost is an ensemble technique that uses multiple weak classifiers to form a powerful binary classification model. It gives more importance to the training instances that are challenging to classify accurately by assigning them different weights.
- `Logistic Regression with L1 or L2 Regularization:` Logistic regression can be regularized with L1 or L2 to enhance model robustness and feature selection, which in turn prevents overfitting.
- `Ensemble Methods (Voting Classifiers):` Voting classifiers use multiple binary classification models such as logistic regression and decision trees to make predictions by taking a majority vote or weighted average of the individual model predictions.

**Summarizing binary classification models:**
| Model                             | Description                                     | Use Cases                                       | Pros                                                           | Cons                                                             |
|-----------------------------------|-------------------------------------------------|-------------------------------------------------|----------------------------------------------------------------|------------------------------------------------------------------|
| Logistic Regression               | Probability-based linear model                  | Healthcare, finance, marketing, text analytics  | Simple, interpretable, works well with small datasets          | May underperform for complex, non-linear relationships             |
| Decision Trees                    | Tree-like structure with binary decisions      | Healthcare, finance, customer churn prediction | Easy to interpret, handles both numeric and categorical data | Prone to overfitting without pruning, may not capture complex patterns |
| Random Forest                     | Ensemble of decision trees                     | Image classification, fraud detection          | Reduces overfitting, high accuracy, handles high-dimensional data | Less interpretable, slower to train compared to individual trees     |
| Support Vector Machines (SVM)     | Finds optimal hyperplane separating classes    | Image classification, bioinformatics          | Effective in high-dimensional spaces, works with non-linear data | Slower training for large datasets, sensitive to kernel choice        |
| Naive Bayes                       | Probabilistic model based on Bayes' theorem    | Text classification, spam detection           | Simple, works well with text data, computationally efficient    | Assumes feature independence (naive assumption), may not capture correlations |
| K-Nearest Neighbors (KNN)         | Classifies based on nearest neighbors         | Recommender systems, pattern recognition       | Intuitive, adapts to local data patterns                        | Sensitive to choice of k, computationally intensive for large datasets |
| Neural Networks                   | Deep learning models with multiple layers      | Image recognition, natural language processing | Can capture complex patterns, state-of-the-art performance      | Requires large datasets, computationally expensive training         |
| Gradient Boosting (XGBoost, LightGBM) | Ensemble methods that boost decision trees | Kaggle competitions, structured data analysis  | High accuracy, handles missing data, good for imbalanced datasets | Tunes hyperparameters, may overfit with insufficient data          |
| Adaboost                          | Boosting ensemble method                       | Face detection, text classification            | Combines weak learners for strong performance                  | Sensitive to noisy data, can overfit if weak classifiers are too complex |
| Logistic Regression with Regularization | Regularized logistic regression           | Feature selection, mitigating overfitting     | Prevents overfitting, automatic feature selection                | Requires tuning of regularization strength, not ideal for very large datasets |
| Ensemble Methods (Voting Classifiers) | Combines multiple classifiers            | General purpose, diverse dataset classification | Improved accuracy through model combination                   | Interpretability may decrease as more models are added              |
----
**Confusion Matrix:**
| Actual / Predicted   | Predicted Positive (P) | Predicted Negative (N) | Total |
|-----------------------|-------------------------|-------------------------|-------|
| Actual Positive (P)  | True Positives (TP)    | False Negatives (FN)   | P     |
| Actual Negative (N)  | False Positives (FP)   | True Negatives (TN)    | N     |
| Total                | P                      | N                      | P + N|

- `True Positives (TP):` The cases in which the model correctly predicted the positive class, and the actual outcome was indeed positive.
- `True Negatives (TN):` The cases in which the model correctly predicted the negative class, and the actual outcome was indeed negative.
- `False Positives (FP):` The cases in which the model incorrectly predicted the positive class (Type I error), but the actual outcome was negative.
- `False Negatives (FN):` The cases in which the model incorrectly predicted the negative class (Type II error), but the actual outcome was positive.

**Evaluating a Binary Classification Model:**
| Metric                         | Formula or Description                | Range                 | Interpretation                                      |
|--------------------------------|---------------------------------------|-----------------------|-----------------------------------------------------|
| Accuracy                        | (TP + TN) / (TP + TN + FP + FN)       | 0 to 1                | Proportion of correct predictions overall.         |
| Precision (Positive Predictive Value) | TP / (TP + FP)                      | 0 to 1                | Proportion of true positives among predicted positives. High precision indicates few false positives.        |
| Recall (Sensitivity or True Positive Rate) | TP / (TP + FN)                | 0 to 1                | Proportion of true positives among actual positives. High recall indicates few false negatives.            |
| F1-Score                        | 2 * (Precision * Recall) / (Precision + Recall) | 0 to 1                | Harmonic mean of precision and recall. Balances precision and recall. |
| Receiver Operating Characteristic (ROC) Curve | Graph of TP rate vs. FP rate       | Area under curve (AUC-ROC) | Measures the model's ability to distinguish between classes. Higher AUC-ROC indicates better discrimination. |
| Precision-Recall (PR) Curve     | Graph of precision vs. recall        | Area under curve (AUC-PR) | Emphasizes performance on positive class. High AUC-PR indicates good precision and recall trade-off.     |
| Area Under the PR Curve (AUC-PR) | Area under the PR curve           | 0 to 1                | Measures the model's ability to balance precision and recall. High AUC-PR is desirable for imbalanced datasets. |
| F-beta Score                    | (1 + beta^2) * (Precision * Recall) / (beta^2 * Precision + Recall) | 0 to 1          | Adjusts the balance between precision and recall using parameter beta. F1-Score is a special case with beta = 1. |
| Specificity                     | TN / (TN + FP)                      | 0 to 1                | Proportion of true negatives among actual negatives. High specificity indicates few false positives.    |
| Matthews Correlation Coefficient (MCC) | (TP * TN - FP * FN) / sqrt((TP + FP)(TP + FN)(TN + FP)(TN + FN)) | -1 to 1     | Measures the correlation between predictions and actual outcomes, correcting for chance.             |
| Kappa Statistic (Cohen's Kappa) | (Observed agreement - Expected agreement) / (1 - Expected agreement) | -1 to 1     | Measures the agreement between predictions and actual outcomes, correcting for chance.              |
| Log-Loss (Logarithmic Loss)    | -1/n * Σ(y log(p) + (1 - y) log(1 - p)) | 0 to ∞              | Measures the accuracy of predicted probabilities for each instance. Lower log-loss indicates better predictions. |

`A high precision model is conservative:` Although it may not always identify the class accurately, once it does, we can have confidence in the accuracy of its response..<br/>
`A high recall model is liberal:` it recognizes a class much more often, but in doing so it tends to include a lot of noise as well (false positives).<br/>
`F1 score:` If our F1 score increases, it means that our model has increased performance for accuracy, recall or both.<br/>
`ROC AUC:` The performance of a model can be measured by a single scalar value called the area under the ROC curve (AUC). A higher value of AUC, which ranges from 0 to 1, indicates better performance. An AUC of 0.5 suggests a random guess, while an AUC of 1.0 indicates flawless classification.