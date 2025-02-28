import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import MinMaxScaler


def default_means(size):
    return 10 * np.random.random(size=size)


def create_2d_test_data(scales, weights, n_clusters, dimensions,
                        min_distance=0, max_distance=np.inf,
                        num_samples=1000,
                        means_func=default_means,
                        normalize=True):

    # Ensures means generated are at least min_distance apart
    # and at most max_distance apart
    def generate_valid_points(n_clusters):
        while True:
            points = means_func(size=(n_clusters, dimensions))
            distances = squareform(pdist(points))

            min_valid = np.all(distances[np.triu_indices(n_clusters, k=1)] >= min_distance)

            max_valid = all(
                np.any(distances[i, np.arange(n_clusters) != i] <= max_distance)
                for i in range(n_clusters)
            )

            if min_valid and max_valid:
                return points

    means = generate_valid_points(n_clusters=n_clusters)

    weights = weights / weights.sum()
    counts = (weights * num_samples).astype(int)

    data = []
    labels = []
    for i, count in enumerate(counts):
        cluster_data = means[i] + np.random.normal(size=(count, dimensions)) * scales[i]
        data.append(cluster_data)

        cluster_labels = np.full(count, i)
        labels.append(cluster_labels)

    data = np.vstack(data)
    labels = np.concatenate(labels)

    if normalize:
        data = MinMaxScaler().fit_transform(data)

    return data, labels
