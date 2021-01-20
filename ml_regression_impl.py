import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# ----------------------------------------
class Regression():

    def score(self, X_test, y_test):
        y_pred = self.predict(X_test)
        return 1 - self.loss(y_test, y_pred) / self.loss(y_test, y_test.mean())

    def fit(self):
        pass
    def predict(self, X):
        pass

# ----------------------------------------
class LinearRegressionLSImp(Regression):

    def __init__(self, learning_rate=0.01, max_iter=100):
        self.learning_rate = learning_rate
        self.max_iter = max_iter

    def loss(self, y, y_pred):
        return np.mean((y - y_pred) ** 2)

    def h(self, X):
        return (X @ self.M) + self.c

    def fit(self, X, y):
        X_mean = X.mean(axis=0)
        y_mean = y.mean()
        self.M = ((X-X_mean).T @ (y-y_mean)) / np.sum((X-X_mean)**2)
        self.c = y_mean - (X_mean @ self.M)

    def predict(self, X):
        return self.h(X)

# ----------------------------------------
class LinearRegressionGDImp(Regression):

    def __init__(self, learning_rate=1, max_iter=100, tol=1e-3):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol

    def loss(self, y, y_pred):
        return np.mean((y - y_pred) ** 2)

    def gradient(self, X, y, y_pred):
        dM = (X.T @ (y_pred - y)) / self.n
        dc = np.mean(y_pred - y)
        return dM, dc

    def h(self, X):
        return (X @ self.M.T) + self.c

    def fit(self, X, y):
        self.n, self.m = X.shape
        self.M, self.c = np.zeros(self.m), 0

        counter = 0
        while counter < self.max_iter:
            y_pred = self.h(X)
            diff_M, diff_c = self.gradient(X, y, y_pred)

            error = self.loss(y, y_pred)
            #print(error)
            if error <= self.tol:
                break

            self.M -= self.learning_rate * diff_M
            self.c -= self.learning_rate * diff_c

            counter += 1

    def predict(self, X):
        return np.round(self.h(X))

def plotLinearRegression(X, y, M, c):
    plt.figure()
    plt.scatter(X[:, 0], y)
    plt.plot(X[:, 0], X[:, 0] * M[0] + c, c='r')
    plt.show()

# ----------------------------------------
# Load dataset
# ----------------------------------------
X, y = make_regression(n_samples=1000, n_features=1, n_informative=1, noise=15)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#X = np.array([[1, 1, 1], [3, 3, 3], [5, 5, 5], [7, 7, 7], [9, 9, 9]])
#y = np.array([20, 52, 45, 85, 80])

#X = np.array([[2], [3], [5], [13], [8], [16], [11], [1], [9]])
#y = np.array([15, 28, 42, 64, 50, 90, 58, 8, 54])

# ----------------------------------------
# Linear Regression
# ----------------------------------------
clf = LinearRegression()
clf.fit(X_train, y_train)
print('Test set score (sklearn) : {:.2f}'.format(clf.score(X_test, y_test)))
print(clf.coef_, clf.intercept_)
plotLinearRegression(X, y, clf.coef_, clf.intercept_)

clf = LinearRegressionLSImp()
clf.fit(X_train, y_train)
print('Test set score (LS) : {:.2f}'.format(clf.score(X_test, y_test)))
print(clf.M, clf.c)
plotLinearRegression(X, y, clf.M, clf.c)

clf = LinearRegressionGDImp(learning_rate=1, max_iter=50000, tol=1e-2)
clf.fit(X_train, y_train)
print('Test set score (GD) : {:.2f}'.format(clf.score(X_test, y_test)))
print(clf.M, clf.c)
plotLinearRegression(X, y, clf.M, clf.c)