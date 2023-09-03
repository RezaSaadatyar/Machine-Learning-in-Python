**Classification**<br/>

**Binary Classification:**<br/>
In machine learning, binary classification is a supervised learning algorithm that categorizes new observations into one of two outcomes usually represented as 0 or 1, true or false, positive or negative, etc.

How Binary Classification Works?

In binary classification, the algorithm is trained on
a labeled dataset, where each data point is
associated with a binary label.
The algorithm then learns to map the input
features to the corresponding binary label. Once
trained, the algorithm can be used to predict the
binary label for new, unseen data points.

Common Binary Classification Models
Logistic Regression: It is used for binary classification problems, where
the output variable is categorical with two
possible values.
Neural Networks: This algorithm is designed to cluster raw input,
recognize patterns, or interpret sensory data.
Despite their multiple advantages, neural
networks require significant computational
resources. It can get complicated to fit a neural
network when there are thousands of
observations.
Support Vector Machines: A support vector machine is typically used for
classification problems by constructing a
hyperplane where the distance between two
classes of data points is at its maximum. This
hyperplane is known as the decision boundary,
separating the classes of data points (e.g.,
oranges vs. apples) on either side of the plane.
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