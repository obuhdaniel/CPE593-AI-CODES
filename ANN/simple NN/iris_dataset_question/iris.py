import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets

from sklearn.model_selection import train_test_split

iris = datasets.load_iris()

df = pd.DataFrame(iris.data, columns= iris.feature_names)
df['target']= iris.target
df['target_name'] = df['target'].map(lambda x: iris.target_names[x])
df = df.iloc[:100, [0,2,4,5]]

#lets convert the categorical class called target names to either 1 or -1
df['target'] = np.where((df['target_name'] =='setosa'), -1, 1)
Y = df['target']
X = df.iloc[:,:2].values
# plt.figure(figsize=(8,6))
# plt.scatter(X[:50, 0], X[:50, 1], marker='o', label='setosa(-1)', color='red')
# plt.scatter(X[50:100, 0], X[50:100, 1], marker='x', label='versicolor(1)', color='blue')
# plt.xlabel('sepal length(cm)')
# plt.ylabel('petal lenght(cm)')
# plt.plot()
# plt.show()


#lets split the data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


class Perceptron:
    def __init__(self, num_features, learning_rate=0.01, n_iter=1000):
        self.num_features = num_features
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.weights = np.zeros((num_features, 1), dtype=float)

        self.bias = np.zeros((1,), dtype=float)
    
    def activation_function(self, x):
        return np.where(x >= 0, 1, -1)

    def forward(self, X):
        linear = np.dot(X, self.weights) + self.bias
        predictions = self.activation_function(linear)
        return predictions
    
    def backward(self, X, y):
        for _ in range(self.n_iter):
            predictions = self.forward(X)
            errors = y.values - predictions.reshape(-1)
            self.weights += self.learning_rate * np.dot(X.T, errors.reshape(-1, 1))
            self.bias += self.learning_rate * np.sum(errors)

    def evaluate(self, X, y):
        predictions = self.forward(X)
        accuracy = np.mean(predictions.flatten() == y.values.flatten())
        return accuracy
    
    def think(self, inputs): 
        inputs = inputs.astype(float)
        output = self.activation_function(np.dot(inputs, self.weights) + self.bias)
        return 'setosa' if output == -1 else 'versicolor'

    


if __name__ == "__main__":
    num_features = X_train.shape[1]
    perceptron = Perceptron(num_features=num_features, learning_rate=0.01, n_iter=10)
    perceptron.backward(X_train, y_train)
    accuracy = perceptron.evaluate(X_test, y_test)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Final weights: {perceptron.weights.flatten()}")
    A = str(input("Sepal Length: "))
    B = str(input("Petal Lenght: "))
    print("New situation: Iris Data = ", A, B)
    print("Output data: ")
    print(perceptron.think(np.array([A, B])))
# The perceptron class is a simple implementation of a single-layer neural network. It uses the perceptron learning algorithm to classify data points into two classes. The class has methods for forward propagation, backward propagation (training), and evaluation of the model's accuracy on test data.
