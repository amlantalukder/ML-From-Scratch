import math, pdb
from mpl_toolkits import mplot3d
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer

# ----------------------------------------
class DecisionTreeClassifierImp():

    def __init__(self, max_depth=10):
        self.max_depth = max_depth
        self.classes = None
        self.tree = None
        self.feature_importances_ = None

    def fit(self, X, y):
        self.n, self.m = X.shape
        self.classes = np.unique(y)
        self.buildTree(X, y)
        self.feature_importances_ = np.zeros(self.m)
        self.calculateFeatureImportances(self.tree)
        self.feature_importances_ /= np.sum(self.feature_importances_)

    def predict(self, X):
        return [self.predictRow(x, self.tree) for x in X]

    def isLeaf(self, node):
        return not isinstance(node, dict)

    def predictRow(self, x, node):
        if self.isLeaf(node):
            return node

        if x[node['index']] < node['value']:
            return self.predictRow(x, node['left'])

        return self.predictRow(x, node['right'])

    def buildTree(self, X, y):
        data = np.concatenate((X, np.expand_dims(y, axis=1)), axis=1)
        root = self.getRoot(data)
        self.tree = self.fork(root)

    def getRoot(self, data):
        root, max_info_gain = None, None
        parent_gini = self.giniIndex(data)
        for index in range(data.shape[1]-1):
            for row in data:
                left, right = self.split(data, index, row[index])
                info_gain = self.infoGain([left, right], parent_gini)
                if max_info_gain == None or max_info_gain < info_gain:
                    max_info_gain = info_gain
                    root = {'index':index, 'value':row[index], 'left':left, 'right':right, 'info_gain':info_gain}
        return root

    def split(self, data, index, val):
        return data[data[:, index]<val], data[data[:, index]>=val]

    def createLeafNode(self, data):
        if len(data) == 0:
            return 0
        classes, freq = np.unique(data[:, -1], return_counts=True)
        return classes[np.argmax(freq)]

    def fork(self, node, depth=1):
        left, right = node['left'], node['right']
        if len(left) == 0 or len(right) == 0:
            node['left'] = node['right'] = self.createLeafNode(np.concatenate((node['left'], node['right']), axis=0))
            return node
        if depth > self.max_depth:
            node['left'], node['right'] = self.createLeafNode(node['left']), self.createLeafNode(node['right'])
            return node

        root = self.getRoot(node['left'])
        node['left'] = self.fork(root, depth + 1)
        root = self.getRoot(node['right'])
        node['right'] = self.fork(root, depth + 1)

        return node

    def giniIndex(self, group):
        gini = 1
        l = len(group)
        if l > 0:
            classes, freq = np.unique(group[:, -1], return_counts=True)
            for i in range(len(classes)):
                p = freq[i]/l
                gini -= p*p
        return gini

    def infoGain(self, child_groups, parent_gini):

        total_rows = sum([len(g) for g in child_groups])

        child_gini = 0
        for g in child_groups:
            child_gini += self.giniIndex(g)*(len(g)/total_rows)

        return parent_gini-child_gini

    def calculateFeatureImportances(self, node):
        if self.isLeaf(node): return
        self.feature_importances_[node['index']] += node['info_gain']
        self.calculateFeatureImportances(node['left'])
        self.calculateFeatureImportances(node['right'])

# ----------------------------------------
class GaussianNBImp():

    def __init__(self):
        self.n = self.m = None
        self.classes = None
        self.mu = self.var = None
        self.class_prior = None

    def getDistribution(self, X):
        mu = np.average(X, axis=0)
        # sigma = np.average((X-mu)**2, axis=0)
        var = np.var(X, axis=0)
        return mu, var

    def fit(self, X, y):
        self.n, self.m = X.shape
        self.classes = np.unique(y)
        self.mu, self.var = [], []
        self.class_prior = []
        for i in range(len(self.classes)):
            X_class = X[y == self.classes[i]]
            self.class_prior.append(X_class.shape[0] / self.n)
            m, s = self.getDistribution(X_class)
            self.mu.append(m)
            self.var.append(s)
        self.mu = np.array(self.mu)
        self.var = np.array(self.var)
        self.class_prior = np.array(self.class_prior)

    def posterior(self, X):
        joint_prob = []
        for i in range(len(self.classes)):
            p = self.class_prior[i]
            p -= 0.5 * np.sum(np.log(2. * np.pi * self.var[i, :]))
            p -= 0.5 * np.sum(((X - self.mu[i, :]) ** 2) / (self.var[i, :]), axis=1)
            joint_prob.append(p)
        return np.array(joint_prob).T

    def predict(self, X):
        return self.classes[np.argmax(self.posterior(X), axis=1)]

# ----------------------------------------
class KNeighborsClassifierImp():

    def __init__(self, n_neighbours=1):
        self.n_neighbours = n_neighbours
        self.X = None
        self.y = None

    def fit(self, X, y):
        self.X, self.y, self.classes = X, y, np.unique(y)

    def getDist(self, x):
        return np.sqrt(np.sum((self.X-x)**2, axis=1))

    def predict(self, X):
        preds = []
        for x in X:
            distances = self.getDist(x)
            votes_nei = self.y[np.argsort(distances)[:self.n_neighbours]]
            max_vote_index = np.argmax(np.bincount(votes_nei))
            preds.append(self.classes[max_vote_index])
        return np.array(preds)

# ----------------------------------------
class LogisticRegressionImp():

    def __init__(self, learning_rate=1, max_iter=100, tol=1e-4):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol

    def loss(self, y, y_pred):
        # ----------------------------------------
        # Due to the use of the logistic function
        # to compute y_pred, the regular least 
        # squared error function,
        # 0.5 * np.mean((y - y_pred) ** 2)
        # becomes a non-convex function and has 
        # multiple local optima. So instead of 
        # this, the following convex function 
        # was used that serves the same purpose.
        #   loss = -log(y_pred) if y = 1 
        #   loss = -log(1-y_pred) if y = 0
        # In short,
        #   loss = -y log(y_pred) - (1-y) log(1-y_pred)
        # ----------------------------------------
        epsilon = 1e-06
        return np.mean(-y @ np.log(y_pred + epsilon) - (1 - y) @ np.log(1 - y_pred + epsilon))

    def gradient(self, X, y, y_pred):
        # ----------------------------------------
        # Since, loss = f(y_pred), y_pred = sig(z)
        # and z = MX + c,
        # d(loss)/dM = d(loss)/d(sig(z)) * d(sig(z))/dz * dz/dM
        # d(loss)/d(sig(z)) = -y/y_pred + (1-y)/(1-y_pred)
        #                   = (-y+y_pred)/(y_pred(1-y_pred))
        # d(sig(z))/dz      = y_pred(1-y_pred)
        # dz/dM             = X
        # So, d(loss)/dM = (y_pred-y)X
        # ----------------------------------------
        dM = (X.T @ (y_pred-y)) / self.n
        dc = np.mean(y_pred-y)
        return dM, dc

    def sigmoid(self, y):
        return 1 / (1 + np.exp(-y))

    def h(self, X):
        z = (X @ self.M.T) + self.c
        return self.sigmoid(z)

    def fit(self, X, y):
        self.n, self.m = X.shape
        self.M, self.c = np.zeros(self.m), 0

        for counter in range(self.max_iter):
            y_pred = self.h(X)
            dM, dc = self.gradient(X, y, y_pred)

            error = self.loss(y, y_pred)
            if error <= self.tol: break

            self.M -= self.learning_rate * dM
            self.c -= self.learning_rate * dc

    def predict(self, X):
        return np.round(self.h(X))

# ----------------------------------------
# Load dataset
# ----------------------------------------
dataset = load_breast_cancer()
X, y = dataset.data, dataset.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
'''
# ----------------------------------------
# Decision Tree
# ----------------------------------------
clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print('Test set accuracy (sklearn) : {}, {:.2f}'.format(np.count_nonzero(y_test == y_pred), np.mean(y_test == y_pred)))
print(clf.feature_importances_)

clf = DecisionTreeClassifierImp()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print('Test set accuracy : {}, {:.2f}'.format(np.count_nonzero(y_test == y_pred), np.mean(y_test == y_pred)))
print(clf.feature_importances_)

# ----------------------------------------
# Naive Bayes
# ----------------------------------------
clf = GaussianNB(var_smoothing=0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print('Test set accuracy (sklearn) : {}, {:.2f}'.format(np.count_nonzero(y_test == y_pred), np.mean(y_test == y_pred)))

clf = GaussianNBImp()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print('Test set accuracy : {}, {:.2f}'.format(np.count_nonzero(y_test == y_pred), np.mean(y_test == y_pred)))

# ----------------------------------------
# KNN
# ----------------------------------------
clf = KNeighborsClassifier(n_neighbors=5)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print('Test set accuracy (sklearn) : {}, {:.2f}'.format(np.count_nonzero(y_test == y_pred), np.mean(y_test == y_pred)))

clf = KNeighborsClassifierImp(n_neighbours=5)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print('Test set accuracy : {}, {:.2f}'.format(np.count_nonzero(y_test == y_pred), np.mean(y_test == y_pred)))
'''
# ----------------------------------------
# Logistic Regression
# ----------------------------------------
clf = LogisticRegression()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print('Test set accuracy (sklearn) : {}, {:.2f}'.format(np.count_nonzero(y_test == y_pred), np.mean(y_test == y_pred)))

clf = LogisticRegressionImp(learning_rate=0.00001, max_iter=10000, tol=1e-2)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print('Test set accuracy : {}, {:.2f}'.format(np.count_nonzero(y_test == y_pred), np.mean(y_test == y_pred)))