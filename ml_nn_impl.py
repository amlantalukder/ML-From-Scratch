import pdb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from keras.models import Sequential
from keras.layers import Dense

class Layer():

    def __init__(self, num_units, input_dim=None):
        self.input_dim = input_dim
        self.num_units = num_units

    def compile(self, input_dim, learning_rate):
        self.learning_rate = learning_rate
        self.M = np.random.rand(input_dim, self.num_units)
        self.c = np.random.rand(self.num_units)

    def gradient(self, extra):
        dM = (self.X.T @ extra) / self.X.shape[0]
        dc = np.mean(extra)
        return dM, dc

    def sigmoid(self, y):
        return 1 / (1 + np.exp(-y))

    def h(self, X):
        z = (X @ self.M) + self.c
        return self.sigmoid(z)

    def forward(self, X):
        self.X = X
        self.y_pred = self.h(X)
        return self.y_pred

    def backward(self, delta_prev):
        activation_derivative = self.y_pred * (1 - self.y_pred)
        dM, dc = self.gradient(delta_prev * activation_derivative)

        delta = (delta_prev * activation_derivative) @ self.M.T

        self.M -= self.learning_rate * dM
        self.c -= self.learning_rate * dc
        return delta

class NeuralNet():

    def __init__(self, layers=None):
        self.layers = []
        if layers:
            for layer in layers:
                self.add(layer)

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, delta):
        for layer in self.layers[::-1]:
            delta = layer.backward(delta)

    def compile(self, learning_rate=1):
        input_dim = self.layers[0].input_dim
        for layer in self.layers:
            layer.compile(input_dim, learning_rate)
            input_dim = layer.num_units

    def fit(self, X, y, epochs=10, batch_size=10):
        for epoch in range(epochs):
            print('Epoch {}/{}'.format(epoch+1, epochs))
            for i in range(0, X.shape[0]-batch_size, batch_size):
                y_pred = self.forward(X[i:i+batch_size])
                y_true = np.expand_dims(y[i:i+batch_size], axis=1)
                delta = 2*(y_pred - y_true)
                print('loss: {:.4f}, accuracy: {:.4f}'.format(np.sum(delta**2), np.mean(y_test == y_true)))
                self.backward(delta)

    def predict(self, X):
        return np.round(self.forward(X))


# ----------------------------------------
# Load dataset
# ----------------------------------------
dataset = load_breast_cancer()
X, y = dataset.data, dataset.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# ----------------------------------------
# Neural network
# ----------------------------------------
model = Sequential()
model.add(Dense(60, input_dim=X_train.shape[1], activation='sigmoid'))
model.add(Dense(30, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=100, batch_size=100)
_, accuracy = model.evaluate(X_test, y_test)
print('Test set accuracy (keras) : {:.2f}'.format(accuracy))


model = NeuralNet()
model.add(Layer(60, input_dim=X_train.shape[1]))
model.add(Layer(30))
model.add(Layer(1))

model.compile(learning_rate=1)
model.fit(X_train, y_train, epochs=100, batch_size=100)
y_pred = model.predict(X_test)
pdb.set_trace()
print('Test set accuracy : {}, {:.2f}'.format(np.count_nonzero(y_test == y_pred), np.mean(y_test == y_pred)))
