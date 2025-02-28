import numpy as np
from scipy.spatial import distance_matrix
from utils import faster_uniquerows
from itertools import combinations
from scipy.spatial import distance

import pandas as pd


def sort_cuts_and_costs(cuts, costs):
    sort_idx = costs.argsort()
    return cuts[:, sort_idx], costs[sort_idx]


def sort_and_discard_cuts(cuts, costs):
    cuts_sorted, costs_sorted = sort_cuts_and_costs(cuts, costs)
    # cuts, costs = remove_duplicate_cuts(cuts_sorted, costs_sorted)
    return cuts_sorted, costs_sorted


class CostFunction:
    """Parent class of all cost functions"""
    def __init__(self):
        raise NotImplementedError

    def compute(self, cuts):
        costs = self.compute_(cuts)
        cuts, costs = sort_and_discard_cuts(cuts, costs)
        return cuts, costs

    def compute_(self, cuts):
        raise NotImplementedError

    def sort_cuts(self, costs, cuts):
        return cuts[:, costs.argsort()]


class DistanceToOtherMean(CostFunction):
    """Linear but terrible."""
    def __init__(self, data):
        self.data = data

    def compute_(self, cuts):
        costs = np.zeros(cuts.shape[1])
        for i, cut in enumerate(cuts.T):
            mean_a = np.mean(self.data[cut == 1], axis=0)
            mean_b = np.mean(self.data[cut == 0], axis=0)

            # Calculate distances to each mean
            dist_to_a = np.linalg.norm(self.data - mean_a, axis=1)
            dist_to_b = np.linalg.norm(self.data - mean_b, axis=1)

            cost_a = np.sum(-dist_to_a[cut == 0])
            cost_b = np.sum(-dist_to_b[cut == 1])

            costs[i] = (cost_a + cost_b)

        return costs


class SimilarityMatrix(CostFunction):
    def __init__(self, data, margin=0.1):
        self.data = data
        self.distance_matrix = distance_matrix(data, data)
        self.distance_matrix[self.distance_matrix > margin] = margin
        self.distance_matrix = 1 - self.distance_matrix / np.max(self.distance_matrix)

    def compute_(self, cuts):
        """For every data point on each side of the cut,
           compute the sum of the distances from all points on one
           side to all points on the other side.
        """
        costs = np.zeros(cuts.shape[1])
        for i, cut in enumerate(cuts.T):
            distances = self.distance_matrix[cut == 0][:, cut == 1]
            costs[i] += (distances.sum())

        return costs


class SumOfDistances(CostFunction):
    def __init__(self, data):
        self.data = data
        self.distance_matrix = distance_matrix(data, data)
        # self.distance_matrix[self.distance_matrix > 1.5] = 1.5
        # self.distance_matrix = 1 - self.distance_matrix / np.max(self.distance_matrix)

    def compute_(self, cuts):
        """For every data point on each side of the cut,
           compute the sum of the distances from all points on one
           side to all points on the other side.
        """
        costs = np.zeros(cuts.shape[1])
        for i, cut in enumerate(cuts.T):
            distances = self.distance_matrix[cut == 0][:, cut == 1]
            costs[i] += (distances.sum())

        return -costs


class OptimizedSimilarityMatrix(CostFunction):
    """Only applicable with equidistant axis-parallel cuts still n²"""
    def __init__(self, data, k=50, margin=1.5):
        self.data = data
        self.dimensions = data.shape[1]
        self.block_combinations = list(combinations(range(k + 1), 2))
        self.k = k
        self.margin = margin

    def compute_(self, cuts):
        costs = np.zeros(cuts.shape[1])
        data = self.data
        i = 0

        for dimension in range(self.dimensions):
            regions = []
            splits = np.linspace(data[:, dimension].min(),
                                 data[:, dimension].max(),
                                 self.k + 2)[1: -1]
            prev_split = self.data[:, dimension].min()

            for split in splits:
                regions.append(data[
                    (data[:, dimension] >= prev_split)
                    & (data[:, dimension] < split)])

                prev_split = split
                i += 1

            regions.append(data[data[:, dimension] >= splits[-1]])

            blocked_distances = np.zeros((self.k + 1, self.k + 1))
            for (a, b) in self.block_combinations:
                if np.abs(splits[a] - splits[b - 1]) > self.margin:
                    cost = 0
                else:
                    distances = distance_matrix(regions[a], regions[b])
                    distances[distances > self.margin] = self.margin
                    distances = 1 - distances / self.margin
                    cost = distances.sum()

                blocked_distances[a, b] = cost

            for a in range(self.k):
                costs[self.k * dimension + a] = blocked_distances[
                    : a + 1, a + 1:].sum()

        return costs


class UltraOptimizedSimilarity(CostFunction):
    """ Not ultra optimized at all.
        Didn't really work.
        Poor attempt to speed up n² distance matrix computation
        There's a lot of overhead, and it's still n², but it's often
        bound by amount of regions instead, which often scales with
        the number of cuts."""
    def __init__(self, data):
        self.data = data
        self.margin = 1.5

    def are_bounding_cubes_within_margin(self, region_a, region_b):
        min_a = np.min(region_a, axis=0)
        max_a = np.max(region_a, axis=0)
        min_b = np.min(region_b, axis=0)
        max_b = np.max(region_b, axis=0)

        # Calculate the shortest distance between the hypercubes
        distances = np.maximum(0, np.maximum(min_a - max_b, min_b - max_a))
        closest_distance = np.sqrt(np.sum(distances ** 2))
        return True if closest_distance < self.margin else False

    def compute_(self, cuts):
        costs = np.zeros(cuts.shape[1])

        unique, assignments = np.unique(cuts, axis=0, return_inverse=True)
        count_regions = len(unique)

        block_combinations = list(combinations(range(count_regions), 2))

        blocked_costs = np.zeros((count_regions, count_regions))
        for a, b in block_combinations:
            region_a = self.data[assignments == a]
            region_b = self.data[assignments == b]
            if self.are_bounding_cubes_within_margin(region_a, region_b):
                distances = distance_matrix(region_a, region_b)
                distances[distances > self.margin] = self.margin
                distances = 1 - distances / self.margin
                blocked_costs[a, b] = distances.sum()

        for i in range(cuts.shape[1]):
            cut = cuts[:, i]
            a = np.unique(assignments[cut == 0])
            b = np.unique(assignments[cut == 1])

            assert set(a).intersection(b) == set()
            costs[i] = blocked_costs[np.ix_(a, b)].sum()

        return costs


class RegionAggregation(CostFunction):
    """
        Algorithm 2 in thesis, with Similarity Matrix similarity measure,
        and mean aggrgation func.

        Essentially the same calculation as sim-matrix,
        but instead of computing distance between all pairs,
        We compute the mean within each "region", and then use those
        as distances.

        O(k * n + k * r^2)
    """
    def __init__(self, data, margin=0.1):
        self.data = data
        self.margin = margin

    def compute_(self, cuts):
        costs = np.zeros(cuts.shape[1])

        # Step 1 in Thesis algorithm 2
        unique, assignments, counts = faster_uniquerows(cuts, return_counts=True, return_inv=True)

        count_regions = len(unique)

        means = np.zeros((count_regions, self.data.shape[1]))

        # Step 2 in Thesis algorithm 2
        for assignment in range(len(counts)):
            region = self.data[assignments == assignment]
            mean = np.mean(region, axis=0)
            means[assignment] = mean

        # Step 3 in Thesis algorithm 2, with similarity matrix function
        pairwise_dists = distance.cdist(means, means, metric='euclidean')
        pairwise_dists[pairwise_dists > self.margin] = self.margin
        normalized_dists = 1 - pairwise_dists / self.margin

        # We normalize by the counts in each region
        region_cost = (np.multiply.outer(counts, counts) *
                       normalized_dists)

        # Remove evreything above diagonal
        # not strictly necessary as it's symmetrical.
        tril_indices = np.tril_indices(region_cost.shape[0], k=0)
        region_cost[tril_indices] = 0

        # Step 4 in Thesis algorithm 2
        for i in range(cuts.shape[1]):
            cut = cuts[:, i]
            a = pd.unique(assignments[cut == 0])
            b = pd.unique(assignments[cut == 1])
            costs[i] = region_cost[np.ix_(a, b)].sum()

        return costs


class BorderDistances(CostFunction):
    """
    # From thesis

    Currently only works with axis-aligned cuts,
    and assumes there are an equal amount of cuts in each dimension.

    For localized costs, it currently hasn't been
    refactored to accomodate. There are two compute methods.
    Swap them if you're not using localized costs for now.
    """
    def __init__(self, data, margin=0.1, localized_costs=False):
        self.n_dimensions = data.shape[1]
        self.data = data
        self.margin = margin / 2
        self.localized_costs = localized_costs

    def compute_(self, cuts):
        # Very unelegant solution for now
        if self.localized_costs:
            return self.compute_2(cuts)
        else:
            return self.compute_1(cuts)

    def compute_1(self, cuts):
        costs = np.empty(cuts.shape[1])
        k = cuts.shape[1] // self.n_dimensions
        i = 0
        for dimension in range(self.n_dimensions):
            for _ in range(k):
                cut = cuts[:, i]
                split = self.data[cut == 1, dimension].max()
                dist_to_line = np.abs(self.data[:, dimension] - split)
                dist_to_line[dist_to_line > self.margin] = self.margin
                sim = 1 - dist_to_line / self.margin
                costs[i] = sim.sum()
                i += 1
        return costs

    def compute_2(self, cuts):
        """FOR LOCALIZED COSTS"""
        costs = np.empty(cuts.shape[1])
        dimensions = self.data.shape[1]

        for i in range(cuts.shape[1]):
            cut = cuts[:, i]

            if np.all(cut == 0) or np.all(cut == 1):
                costs[i] = 0
                continue
            # Adds d complexity to cost-recomputation in this simple implementation..
            # Can easily be done without, but would need to refactor tangles to maintain
            # the axis of the original cut.
            for axis in range(dimensions):
                X_0 = self.data[cut == 0]
                X_1 = self.data[cut == 1]
                if np.all(X_1[:, axis] > np.max(X_0[:, axis])
                          ) or np.all(X_1[:, axis] < np.min(X_0[:, axis])):
                    cut_axis = axis
                    break

            split = self.data[cut == 1, cut_axis].max()

            dist_to_line = np.abs(self.data[:, cut_axis] - split)
            dist_to_line[dist_to_line > self.margin] = self.margin
            sim = 1 - dist_to_line / self.margin
            costs[i] = sim.sum()

        return costs


class Knn(CostFunction):
    # Terrible
    def __init__(self, data):
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=10)
        nn.fit(data)
        self.distances, self.indices = nn.kneighbors(data)
        self.data = data

    def compute_(self, cuts):
        costs = np.zeros(cuts.shape[1])
        for i, cut in enumerate(cuts.T):
            data_a = self.indices[cut == 1]
            data_b = self.indices[cut == 0]
            boundary_crossings = 0

            indices_a = set(np.where(cut == 1)[0])
            indices_b = set(np.where(cut == 0)[0])
            count_across = 0
            for j in data_a:
                if count := len(set(j).intersection(indices_b)):
                    boundary_crossings += count
                    count_across += 1

            for j in data_b:
                if count := len(set(j).intersection(indices_a)):
                    boundary_crossings += count
                    count_across += 1
            costs[i] = boundary_crossings / (min(len(indices_a), len(indices_b)) + 0.0001)

        return costs


class MinimalVariance(CostFunction):
    # Terrible
    def __init__(self, data):
        self.data = data

    def compute_variance(self, sub_data):
        mean = np.mean(sub_data, axis=0)
        squared_distances = np.linalg.norm(sub_data - mean, axis=1) ** 2
        variance = np.mean(squared_distances)

        return variance

    def compute_(self, cuts):
        costs = np.zeros(cuts.shape[1])
        for i, cut in enumerate(cuts.T):
            data_a = self.data[cut == 1]
            data_b = self.data[cut == 0]
            var_a = self.compute_variance(data_a)
            var_b = self.compute_variance(data_b)

            cost = (len(data_a) * var_a + len(data_b) * var_b)
            costs[i] = cost

        return costs
