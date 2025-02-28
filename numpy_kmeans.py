import numpy as np


class NumpyKMeans:
    def __init__(self, n_clusters=3, max_iter=300, tol=0.0005):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tolerance = tol
        self.centroids = None

    def _init_centroids(self, X):
        n_samples = X.shape[0]
        centroids = [X[np.random.choice(n_samples)]]

        for _ in range(1, self.n_clusters):
            distances = np.min(np.linalg.norm(X[:, np.newaxis] - centroids, axis=2), axis=1)
            probs = distances**2 / np.sum(distances**2)
            centroid = X[np.random.choice(n_samples, p=probs)]
            centroids.append(centroid)

        return np.array(centroids)

    def fit(self, X):

        self.centroids = self._init_centroids(X)

        for _ in range(self.max_iter):
            distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
            labels = np.argmin(distances, axis=1)

            new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(self.n_clusters)])

            if np.linalg.norm(new_centroids - self.centroids) < self.tolerance:
                break

            self.centroids = new_centroids

        self.labels_ = labels

    def predict(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

    def fit_predict(self, X):
        self.fit(X)
        return self.predict(X)
