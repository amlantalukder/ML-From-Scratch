import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import scipy.stats as stats

class KMeansImp():

    def __init__(self, n_clusters=2, max_iter=300, tol=1e-4, random_state=0):
        self.n_clusters = n_clusters
        self.labels = None
        self.cluster_centroids = None
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = 0

    def distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def distanceFromCentroids(self, x):
        return [self.distance(x, c) for c in self.cluster_centroids]

    def adjustCentroid(self, X):
        X_c = []
        for j in range(self.cluster_centroids.shape[0]):
            temp = []
            for i in range(self.labels.shape[0]):
                if self.labels[i] == j:
                    temp.append(X[i])
            X_c.append(np.average(temp, axis=0))
        return np.array(X_c)

    def getRandomCentroids(self, X):
        random_state = np.random.RandomState(self.random_state)
        seeds = random_state.permutation(X.shape[0])[:self.n_clusters]
        return X[seeds]

    def fit(self, X):
        self.cluster_centroids = self.getRandomCentroids(X)

        for _ in range(self.max_iter):
            self.labels = np.array([np.argmin(self.distanceFromCentroids(X[i])) for i in range(X.shape[0])])

            new_centroids = self.adjustCentroid(X)

            converged = True
            for i in range(self.cluster_centroids.shape[0]):
                if self.distance(self.cluster_centroids[i], new_centroids[i]) > self.tol:
                    converged = False

            if converged: break

    def predict(self, X):
        return [np.argmin(self.distanceFromCentroids(X[i])) for i in range(X.shape[0])]

    def plot(X, labels, centroids):
        random.seed(0)
        colors = color = ["#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in
                          range(np.size(centroids))]
        plt.figure()
        for i in range(X.shape[0]):
            plt.scatter(X[i, 0], X[i, 1], c=colors[labels[i]])
        for i in range(centroids.shape[0]):
            plt.scatter(centroids[i, 0], centroids[i, 1], marker='+', c=colors[i])
        plt.show()

class EMGaussian:

    def __init__(self, n_clusters=2, steps=None, tol=0.1, random_state=0):
        self.count = 0
        self.steps = None
        self.records = None
        self.n_clusters = n_clusters
        self.tol = tol
        self.random_state = random_state

    def checkConvergence(self, a, b):
        for i in range(len(a)):
            for j in range(len(a[i])):
                if np.linalg.norm(a[i][j]-b[i][j]) > self.tol:
                    return False
        return True

    def fit(self, X):
        random_state = np.random.RandomState(self.random_state)
        indices = random_state.permutation(X.shape[0])[:self.n_clusters]
        mus = X[indices, :]
        sigmas = np.ones((self.n_clusters, X.shape[1]))

        info = []
        print('Init')
        for i in range(self.n_clusters):
            print('mu: {}, sigma: {}'.format(mus[i], sigmas[i]))
            info.append((mus[i], sigmas[i]))

        self.records = [info]
        self.priors = np.full((self.n_clusters, X.shape[1]), 1/self.n_clusters)
        self.posteriors = np.full((self.n_clusters, X.shape[0], X.shape[1]), None)

        while (self.steps != None and self.count < self.steps) or \
                len(self.records) < 2 or not self.checkConvergence(self.records[-2], self.records[-1]):
            self.eStep(X)
            self.mStep(X)
            self.count += 1
            print('Step {}:'.format(self.count))
            for i in range(len(self.records[-1])):
                print('mu: {}, sigma: {}'.format(self.records[-1][i][0], self.records[-1][i][1]))

        self.plot(X, num=6)

    def likelihood(self, X, mu, sigma):
        try:
            return np.exp((-(X-mu)**2/(2*sigma**2)).astype('float'))/np.sqrt((2*np.pi*sigma**2).astype('float'))
        except:
            pdb.set_trace()

    def eStep(self, X):
        likelihoods = []
        posterior_denom = 0
        for i in range(self.n_clusters):
            mu, sigma = self.records[-1][i]
            l = self.likelihood(X, mu, sigma)
            likelihoods.append(l)
            posterior_denom += l*self.priors[i]

        for i in range(self.n_clusters):
            self.posteriors[i] = np.divide(likelihoods[i]*self.priors[i], posterior_denom, out=np.zeros_like(likelihoods[i]), where=(likelihoods[i]!=0))

    def mStep(self, X):
        info = []
        for i in range(self.n_clusters):
            self.priors[i] = np.mean(self.posteriors[i])
            mu = np.average(X, weights=self.posteriors[i], axis=0)
            sigma = np.average((X-mu)**2, weights=self.posteriors[i], axis=0)

            info.append((mu, sigma))
        self.records.append(info)

    def plot(self, X, num=6):
        if len(self.records) <= 6:
            indices = range(0, len(self.records))
        else:
            indices = [0] + sorted(np.random.choice(len(self.records)-2, size=4, replace=False)+1) + [len(self.records)-1]

        fig, axes = plt.subplots(ncols=3, nrows=len(indices)//3+1*int(len(indices)%3!=0))
        for i in range(len(indices)):
            if len(indices) <= 3:
                ax = axes[i]
            else:
                try:
                    ax = axes[i//3, i%3]
                except:
                    pdb.set_trace()
            info = self.records[indices[i]]
            for j in range(len(info)):
                mu, sigma = info[j]
                x = np.linspace(mu[0] - 3 * sigma[0], mu[0] + 3 * sigma[0], 100)
                ax.plot(x, stats.norm.pdf(x, mu[0], sigma[0]), c='r')
            ax.set_title(indices[i])
            ax.scatter(X[:, 0], [0]*X.shape[0])
            ax.set_ylim(0, 1)
        plt.subplots_adjust(hspace=0.5)
        plt.show()

# ----------------------------------------
# Load dataset
# ----------------------------------------
dataset = load_breast_cancer()
X, y = dataset.data, dataset.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# ----------------------------------------
# Kmeans
# ----------------------------------------
clf = KMeans(n_clusters=2, init='random', random_state=0)
clf.fit(X_train)
y_pred = clf.predict(X_test)
print(clf.cluster_centers_[:, :2])
KMeansImp.plot(X_test, clf.labels_, clf.cluster_centers_)

clf = KMeansImp(n_clusters=2, random_state=0)
clf.fit(X_train)
y_pred = clf.predict(X_test)
print(clf.cluster_centroids[:, :2])
KMeansImp.plot(X_test, clf.labels, clf.cluster_centroids)
print('....')

# ----------------------------------------
# Expectation Maximization
# ----------------------------------------
X = np.concatenate((np.random.normal(1, 1, (100, 3)), np.random.normal(200, 5, (1000, 3))))
clf = EMGaussian(n_clusters=2)
clf.fit(X)