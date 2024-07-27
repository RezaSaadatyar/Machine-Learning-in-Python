import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors

# =================================================== KNN_optimal ========================================================  
def knn_optimal(data_train, label_train, data_test, label_test, display_optimal_k="off", n=21, fig_size=(3.5,2.5)):
    if np.shape(data_train)[0] < np.shape(data_train)[1]:  # Convert Data training & Test >>>> m*n; m > n
        data_train = data_train.T
        data_test = data_test.T
    t = np.arange(1, n)
    accuracy_train = np.zeros(n-1)
    accuracy_test = np.zeros(n-1)
    for i in range(1 , n):
        model = neighbors.KNeighborsClassifier(metric='minkowski', n_neighbors=i)
        model.fit(data_train, label_train)
        accuracy_train[i-1] = model.score(data_train, label_train)
        accuracy_test[i-1] = model.score(data_test, label_test)
    
    if display_optimal_k == "on":
        
        plt.figure(figsize=fig_size)
        plt.plot(t, accuracy_train, label="Training")
        plt.plot(t, accuracy_test, label="Test")
        plt.xticks(t)
        plt.legend(fontsize=8)
        
        plt.xlabel("Number of neighbors")
        plt.ylabel("Accuracy")
        plt.title(f"Optimal_k for KNN: {t[np.argmax(accuracy_test)]}", fontsize=10)
        plt.tick_params(axis='x', rotation=90)

    return t[np.argmax(accuracy_test)]