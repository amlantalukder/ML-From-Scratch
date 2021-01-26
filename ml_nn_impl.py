import pdb
import numpy as np
import matplotlib.pyplot as plt
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
        self.M, self.c = np.zeros((input_dim, self.num_units)), np.zeros(self.num_units)
        #self.M = np.random.rand(input_dim, self.num_units)
        #self.c = np.random.rand(self.num_units)

    def gradient(self, dloss_dz):
        dz_dM, dz_dc = self.X, 1
        dloss_dM = (dz_dM.T @ dloss_dz) / self.X.shape[0]
        dloss_dc = np.mean(dz_dc * dloss_dz, axis=0)
        return dloss_dM, dloss_dc

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def h(self, X):
        z = (X @ self.M) + self.c
        return self.sigmoid(z)

    def forward(self, X):
        self.X = X
        self.y_pred = self.h(X)
        return self.y_pred

    def backward(self, dloss_dy=[], dloss_dz=[]):
        if dloss_dy != []:
            dy_dz = self.y_pred * (1 - self.y_pred)
            dloss_dz = dy_dz * dloss_dy

        dM, dc = self.gradient(dloss_dz)
        
        dz_dX = self.M
        dloss_dX = (dloss_dz) @ dz_dX.T
        #print('X:', self.X[0])
        #print('y_pred:', self.y_pred[:10])
        #print('M:', self.M[:10])
        #print('c:', self.c)
        #print('\n')
        self.M -= self.learning_rate * dM
        self.c -= self.learning_rate * dc
        
        return dloss_dX

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
            #print(X[:2, :2])
            X = layer.forward(X)
            #print(X[:2, :2])
        return X

    def backward(self, delta):
        num_layers = len(self.layers)
        delta = self.layers[num_layers-1].backward(dloss_dz = delta)
        for i in range(num_layers-2, -1, -1):
            delta = self.layers[i].backward(dloss_dy = delta)

    def getLoss(self, y, y_pred):
        if self.loss == 'mean_squared_error':
            return 0.5 * np.mean((y_pred-y)**2)
        return None

    def dloss_dy(self, y, y_pred):
        if self.loss == 'mean_squared_error':
            return (y_pred-y)
        return None

    def compile(self, loss='mean_squared_error', learning_rate=1, verbose=1):
        self.loss = loss
        self.verbose = verbose
        input_dim = self.layers[0].input_dim
        for layer in self.layers:
            layer.compile(input_dim, learning_rate)
            input_dim = layer.num_units

    def fit(self, X, y, epochs=10, batch_size=10):
        history = {'loss':[], 'accuracy':[]}
        for epoch in range(epochs):
            if self.verbose: print('Epoch {}/{}'.format(epoch+1, epochs))
            history_per_epoch = []
            for i in range(0, X.shape[0]-batch_size+1, batch_size):
                
                y_pred = self.forward(X[i:i+batch_size])

                y_true = np.expand_dims(y[i:i+batch_size], axis=1)
                loss, accuracy = self.getLoss(y_true, y_pred), np.mean(np.round(y_pred) == y_true)
                history_per_epoch.append([loss, accuracy])
                if self.verbose: print('loss: {:.4f}, accuracy: {:.4f}'.format(loss, accuracy))
                
                self.backward(self.dloss_dy(y_true, y_pred))
            
            history_per_epoch = np.mean(history_per_epoch, axis=0)
            history['loss'].append(history_per_epoch[0])
            history['accuracy'].append(history_per_epoch[1])

        return history

    def predict(self, X):
        return np.squeeze(np.round(self.forward(X)), axis=1)

# ----------------------------------------
# Show model performance
# ----------------------------------------
def plotPerformance(history):

    PLOT_TITLE_FONT_SIZE = 30
    PLOT_LABEL_FONT_SIZE = 25
    PLOT_LEGEND_FONT_SIZE = 20
    PLOT_TICKLABEL_FONT_SIZE = 15

    plt.figure(figsize=(12, 10))

    plt.plot(history['loss'], label='Loss', c='red')
    plt.plot(history['accuracy'], label='Accuracy', c='blue')

    plt.title('Model Performance', fontsize=PLOT_TITLE_FONT_SIZE)
    plt.xlabel('Epoch', fontsize=PLOT_LABEL_FONT_SIZE)
    plt.ylabel('Performance', fontsize=PLOT_LABEL_FONT_SIZE)
    plt.legend(ncol=2, fontsize=PLOT_LEGEND_FONT_SIZE)
    plt.xticks(fontsize=PLOT_TICKLABEL_FONT_SIZE)
    plt.yticks(fontsize=PLOT_TICKLABEL_FONT_SIZE)

    plt.show()

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

#plotPerformance(history.history)

model = NeuralNet()
model.add(Layer(1, input_dim=X_train.shape[1]))
#model.add(Layer(60, input_dim=X_train.shape[1]))
#model.add(Layer(30))
#model.add(Layer(1))

model.compile(learning_rate=0.00001, verbose=0)
history = model.fit(X_train, y_train, epochs=2000, batch_size=len(X_train))
y_pred = model.predict(X_test)
print('Test set accuracy : {}, {:.2f}'.format(np.count_nonzero(y_test == y_pred), np.mean(y_test == y_pred)))

#plotPerformance(history)