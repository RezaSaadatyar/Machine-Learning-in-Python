**Classification**<br/>

**Binary Classification:**<br/>
It is a type of supervised learning algorithm used in machine learning. It aims to classify new observations into one of two possible outcomes, typically denoted as 0 or 1, true or false, positive or negative, and so on.

**How Binary Classification Works?**<br/>The process of binary classification involves training an algorithm on a dataset that has labeled data points with binary labels. The algorithm then learns how to associate input features with their respective binary labels. Once the algorithm is trained, it can be utilized to predict binary labels for new and unseen data points.

**Common Binary Classification Models:**<br/>
- `Logistic Regression:` This is utilized in problems involving binary classification, where the output variable has two categorical values.
- `Neural Networks:` This algorithm clusters input, recognizes patterns, and interprets sensory data. However, neural networks require significant computational resources, making it difficult to fit them when dealing with thousands of observations.
 - `Support Vector Machines: `Support vector machines are used for classification by creating a hyperplane that maximizes the distance between two classes of data points. This hyperplane, known as the decision boundary, separates the classes of data points on either side. For example, it could separate oranges from apples.
Random Forest: Random forest is another flexible supervised
machine learning algorithm used for both
classification and regression purposes. It is an
ensemble learning algorithm that combines
multiple decision trees to improve accuracy and
reduce overfitting.
Naive Bayes: Naive Bayes assumes that the features (input
variables) are conditionally independent of each
other given the class label. This is a "naive"
assumption because in reality, features may be
correlated with each other. The three main types
of Naive Bayes algorithms: Gaussian Naive Bayes,
Multinomial Naive Bayes and Bernoulli Naive
Bayes.

Evaluating a Binary Classification Model:
True Positive (TP) is when the patient is diseased
and the model predicts "diseased"
False Positive (FP) or Type 1 Error is when the
patient is healthy but the model predicts
"diseased"
True Negative (TN) is when the patient is healthy
and the model predicts "healthy"
False Negative (FN) or Type 2 Error is when the
patient is diseased and the model predicts
"healthy".

Impact of False Negatives and False Positives:
False negatives and false positives can have
different impacts depending on the specific
problem and context of the classification model.
In a medical diagnosis scenario, a false negative
can result in a patient not receiving the necessary
treatment for a disease, leading to a worsened
health condition.
In airport security screening, a false positive
result for a potential threat can result in
unnecessary delays and inconvenience for the
passengers.