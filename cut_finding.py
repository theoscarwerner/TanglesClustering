import numpy as np
from sklearn.cluster import KMeans


def axis_parallel_cuts_range_axis_parallel(data, k=50):
    """
    Partition data into k equal-sized groups based on the number of data points along x and y axes.

    Args:
        data: A 2D numpy array where rows are data points, and columns are x and y values.
        k: Number of partitions per axis.

    Returns:
        A binary matrix indicating the cuts along x and y axes.
    """
    cuts = np.zeros((data.shape[0], 2 * k), dtype=np.int8)

    sorted_x_indices = np.argsort(data[:, 0])
    for i in range(1, k+1):
        threshold_index = int(i * data.shape[0] / (k+1))
        threshold_value = data[sorted_x_indices[threshold_index - 1], 0]
        cuts[data[:, 0] < threshold_value, i - 1] = 1

    sorted_y_indices = np.argsort(data[:, 1])
    for i in range(1, k+1):
        threshold_index = int(i * data.shape[0] / (k+1))
        threshold_value = data[sorted_y_indices[threshold_index - 1], 1]
        cuts[data[:, 1] < threshold_value, k + i - 1] = 1

    return cuts


def projected_1d_two_means(data, k=50):
    dimensions = data.shape[1]

    cuts = np.zeros((data.shape[0], dimensions * k), dtype=np.int8)

    for i in range(dimensions * k):
        random_projection = np.random.randn(data.shape[1])
        projected_data = np.dot(data, random_projection)

        # Apply k-means algorithm with k=2
        kmeans = KMeans(n_clusters=2)
        kmeans.fit(projected_data.reshape(-1, 1))
        labels = kmeans.labels_
        cuts[:, i] = labels

    return cuts


def range_axis_parallel(data, agreement_param=100):
    if agreement_param < 1:
        agreement_param = int(np.floor(
            data.shape[0] * agreement_param))

    assert agreement_param > 1

    cuts = []
    dimensions = data.shape[1]
    length = data.shape[0]

    for dimension in range(dimensions):
        points_one_side = int(agreement_param - 1)
        sorted_indices = np.argsort(data[:, dimension])

        while True:
            if points_one_side >= len(data):
                break
            cut = np.zeros(length, dtype=np.int8)

            cut[sorted_indices[:points_one_side]] = 1
            cuts.append(cut)
            points_one_side += agreement_param - 1
            points_one_side = int(points_one_side)

    return np.array(cuts).T


def axis_parallel_cuts(data, k=50):
    dimensions = data.shape[1]
    cuts = np.zeros((data.shape[0], dimensions*k), dtype=np.int8)
    i = 0
    for dimension in range(dimensions):
        splits = np.linspace(data[:, dimension].min(),
                             data[:, dimension].max(), k+2)[1:-1]
        for split in splits:
            cuts[data[:, dimension] < split, i] = 1
            i += 1

    return cuts


def two_means(data, return_costs=False, k=50):
    """After running axis_parallel_cuts, this function computes the mean of each cluster,
       and reassigns points to the closest of the two centers
     """
    cuts = axis_parallel_cuts(data, k)
    costs = np.zeros(cuts.shape[1])
    for i, cut in enumerate(cuts.T):
        mean_a = np.mean(data[cut == 1], axis=0)
        mean_b = np.mean(data[cut == 0], axis=0)

        # Calculate distances to each mean
        dist_to_a = np.linalg.norm(data - mean_a, axis=1)
        dist_to_b = np.linalg.norm(data - mean_b, axis=1)

        cut = np.where(dist_to_a < dist_to_b, 1, 0)
        cuts[:, i] = cut

        if return_costs:
            costs[i] = (np.sum(np.exp(-dist_to_a[cut == 0])) +
                        np.sum(np.exp(-dist_to_b[cut == 1])))

    if return_costs:
        return cuts, costs

    return cuts


def local_two_means(data, return_costs=False, k=50):
    """After running axis_parallel_cuts, this function computes the mean of each cluster,
       and reassigns points to the closest of the two centers
     """

    dimensions = data.shape[1]

    cuts = np.zeros((data.shape[0], dimensions*k), dtype=np.int8)
    costs = np.zeros(cuts.shape[1])

    i = 0

    for dimension in range(dimensions):
        splits = np.linspace(data[:, dimension].min(),
                             data[:, dimension].max(), k+2)

        for j in range(len(splits) - 2):
            left_split = splits[j]
            middle_split = splits[j+1]
            right_split = splits[j+2]

            cuts[data[:, dimension] >= middle_split, i] = 1

            both_interval_mask = (data[:, dimension] >= left_split) & (
                data[:, dimension] < right_split)

            left_interval = data[(data[:, dimension] >= left_split) &
                                 (data[:, dimension] < middle_split)]
            right_interval = data[(data[:, dimension] >= middle_split) &
                                  (data[:, dimension] < right_split)]

            if len(left_interval) == 0 or len(right_interval) == 0:
                i += 1
                continue

            mean_left = np.mean(left_interval, axis=0)
            mean_right = np.mean(right_interval, axis=0)

            middle_data = data[both_interval_mask]

            dist_to_left = np.linalg.norm(middle_data - mean_left, axis=1)
            dist_to_right = np.linalg.norm(middle_data - mean_right, axis=1)

            cuts[both_interval_mask, i] = np.where(
                dist_to_left < dist_to_right, 0, 1)

            if return_costs:
                costs[i] = (np.sum(
                    np.exp(-dist_to_left[dist_to_left > dist_to_right]))
                    + np.sum(np.exp(-dist_to_right[dist_to_right > dist_to_left])))
            i += 1

    if return_costs:
        from cost_functions import sort_and_discard_cuts
        return sort_and_discard_cuts(cuts, costs)

    return cuts


def local_two_counts(data, return_costs=False, k=50):
    """After running axis_parallel_cuts, this function computes the mean of each cluster,
       and reassigns points to the closest of the two centers
     """
    dimensions = data.shape[1]

    cuts = np.zeros((data.shape[0], dimensions*k), dtype=np.int8)
    costs = np.zeros(cuts.shape[1])

    i = 0

    for dimension in range(dimensions):
        splits = np.linspace(data[:, dimension].min(),
                             data[:, dimension].max(), k+2)

        for j in range(len(splits) - 2):
            left_split = splits[j]
            middle_split = splits[j+1]
            right_split = splits[j+2]

            cuts[data[:, dimension] >= middle_split, i] = 1

            both_interval_mask = (data[:, dimension] >= left_split) & (
                data[:, dimension] < right_split)

            left_interval = data[(data[:, dimension] >= left_split) &
                                 (data[:, dimension] < middle_split)]
            right_interval = data[(data[:, dimension] >= middle_split) &
                                  (data[:, dimension] < right_split)]

            if len(left_interval) == 0 or len(right_interval) == 0:
                i += 1
                continue

            mean_left = np.mean(left_interval, axis=0)
            mean_right = np.mean(right_interval, axis=0)

            middle_data = data[both_interval_mask]

            dist_to_left = np.linalg.norm(middle_data - mean_left, axis=1)
            dist_to_right = np.linalg.norm(middle_data - mean_right, axis=1)

            cuts[both_interval_mask, i] = np.where(
                dist_to_left < dist_to_right, 0, 1)

            if return_costs:
                costs[i] = sum(both_interval_mask)
            i += 1

    if return_costs:
        from cost_functions import sort_and_discard_cuts
        return sort_and_discard_cuts(cuts, costs)

    return cuts
