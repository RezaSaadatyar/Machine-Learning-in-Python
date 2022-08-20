import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors
def KNN_Optimal(Data_Train, Label_Train, Data_Test, Label_Test, N):
    if np.shape(Data_Train)[0] < np.shape(Data_Train)[1]:  # Convert Data training & Test >>>> m*n; m > n
        Data_Train = Data_Train.T
        Data_Test = Data_Test.T
    n = np.arange(1, N)
    Accuracy_Train = np.zeros(N-1)
    Accuracy_Test = np.zeros(N-1)
    for i in range(1 , N):
        model = neighbors.KNeighborsClassifier(metric='minkowski', n_neighbors=i)
        model.fit(Data_Train, Label_Train)
        Accuracy_Train[i-1] = model.score(Data_Train, Label_Train)
        Accuracy_Test[i-1] = model.score(Data_Test, Label_Test)
    plt.plot(n, Accuracy_Train, label=" Training Accuracy")
    plt.plot(n, Accuracy_Test, label=" Test Accuracy")
    plt.xlabel("Number of neighbors")
    plt.ylabel("Accuracy")
    plt.xticks(n)
    plt.legend()
    plt.title("KNN")
    plt.show()
    return n[np.argmax(Accuracy_Test)]

