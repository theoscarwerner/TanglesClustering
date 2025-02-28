import numpy as np
import matplotlib.pyplot as plt

from binarytree import Node
from itertools import combinations
from sklearn.preprocessing import MinMaxScaler
import plotting
from scipy.signal import find_peaks


def normalize(array):
    """
    Uniformly Normalize array to between 0-1
    """
    ptp = np.ptp(array)
    if ptp != 0:
        return (array - np.min(array)) / np.ptp(array)
    else:
        return np.ones_like(array)


class TangleNode(Node):
    def __init__(self,  name, core, is_root=False):
        super().__init__(name)

        self.core = core
        self.name = name
        self.distinguished_cuts = set([])
        self.condensed_orientations = set([])
        self.is_root = is_root
        self.left = None
        self.right = None

    def get_leaves(self, print_=False):
        if self is None:
            return []

        if print_:
            print(self.name)
            print('\tDist:', self.distinguished_cuts)
            print()

        if self.left is None and self.right is None:
            return [self]

        if self.left is None:
            return self.right.get_leaves(print_=print_)

        if self.right is None:
            return self.left.get_leaves(print_=print_)

        return self.left.get_leaves(print_=print_) + self.right.get_leaves(print_=print_)

    def is_leaf(self):
        return self.left is None and self.right is None

    def get_core(self):
        return np.bitwise_and.reduce(self.core, axis=0)


def prune_tree(root, prune_depth):
    def helper(node, depth_from_split):
        if not node:
            return None, depth_from_split

        left_pruned, _ = helper(
            node.left, 0 if node.left and node.right else depth_from_split + 1)
        right_pruned, _ = helper(
            node.right, 0 if node.left and node.right else depth_from_split + 1)

        node.left, node.right = left_pruned, right_pruned

        if not node.left and not node.right and depth_from_split < prune_depth:
            return None, depth_from_split

        return node, depth_from_split

    new_root, _ = helper(root, 0)
    return new_root


class TangleClustering:
    def __init__(self,
                 cut_function=None,
                 cuts=None,
                 cost_class=None,
                 costs=None,
                 agreement_param=0.1,
                 n_clusters=None,
                 prune_from_projection=False,
                 recompute_costs_on_split=False,
                 prune_depth=None,
                 plot_hist=False
                 ):
        # n_clusters is just to make it easier to run in loops when comparing
        # with sklearn models.
        # It's not actually used anywhere in the code.
        if cost_class is None and costs is None:
            raise ValueError('Either cost_class or costs must be provided')

        if cost_class is not None and costs is not None:
            raise ValueError('Either cost_class or costs must be provided, not both')

        if cuts is None and cut_function is None:
            raise ValueError('Either cuts or cut_function must be provided')

        if cuts is not None and cut_function is not None:
            raise ValueError('Either cuts or cut_function must be provided, not both')

        if recompute_costs_on_split and cost_class is None:
            raise ValueError('Cost class must be provided to recompute costs on split')

        if not isinstance(prune_depth, int) and prune_depth is not None:
            raise ValueError('Prune depth must be an integer')

        self.root = None
        self.cut_function = cut_function
        self.cuts = cuts
        self.CostClass = cost_class
        self.costs = costs
        self.agreement_param = agreement_param
        self.contracted = False
        self.prune_from_projection = prune_from_projection
        self.recompute_costs_on_split = recompute_costs_on_split
        self.prune_depth = prune_depth
        self.plot_hist = plot_hist

    def condense_tree(self, root):
        """
        Removes all nodes with exactly one child
        Additionally, stores the name of the current node and all parent nodes
        until the most recent splitting node (node with two children).
        """
        if root is None:
            return None

        def process_node(node, parent_names):
            if node is None:
                return None

            current_names = parent_names | set([node.value])

            if node.left is None and node.right is None:
                node.condensed_orientations |= current_names
                return node

            if node.left is None:
                child = process_node(node.right, current_names)
                return child

            if node.right is None:
                child = process_node(node.left, current_names)
                return child

            node.left = process_node(node.left, set([]))
            node.right = process_node(node.right, set([]))
            node.condensed_orientations = current_names
            return node

        root = process_node(root, set([]))
        root.is_root = True
        # Start processing the tree from the root
        return root

    def contract_tree(self, node):
        if not node.is_leaf():
            self.contract_tree(node.left)
            self.contract_tree(node.right)
            for condensed_orientation in node.left.condensed_orientations:
                if condensed_orientation in node.right.condensed_orientations:
                    node.condensed_orientations.add(condensed_orientation)
                elif -condensed_orientation in node.right.condensed_orientations:
                    node.distinguished_cuts.add(condensed_orientation)
        self.contracted = True

    def is_consistent(self, core, min_agreement_size, assignment):

        if len(core) == 0:
            if assignment.sum() < min_agreement_size:
                return False

        elif len(core) == 1:
            if (core[0] & assignment).sum() < min_agreement_size:
                return False
        else:
            for core1, core2 in combinations(core, 2):
                if (core1 & core2 & assignment).sum() < min_agreement_size:
                    return False

        return True

    def get_most_restrictive_core(self, core, A):

        # If A does not restrict the core further, then we just reuse the core
        if np.any(np.all((core == 0) | (A == core), axis=1)):
            return core

        # if A is a subset of a row in Core, it is more restrictive.
        # We remove the non-restrcting cuts.
        mask = np.all((A == 0) | (A == core), axis=1)
        core = core[~mask]

        core_new = np.vstack([core, A])
        return core_new

    def compute_tangles(self):
        # Algorithm 1 in the thesis

        self.root = TangleNode(
            'root', core=np.empty((0, self.cuts.shape[0]), dtype=np.int8),
            is_root=True)

        current_leaves = [self.root]

        # If agreement_param < 1, we assume it's passed
        # as a fraction of total data size.
        if self.agreement_param < 1:
            min_agreement_size = self.agreement_param * self.cuts.shape[0]
        else:
            min_agreement_size = self.agreement_param

        for cut_index in range(self.cuts.shape[1]):
            cut = self.cuts[:, cut_index]
            new_leaves = []
            # Loop over all new leaves (leaf is active if it may still be possible to add child)
            for current_leaf in current_leaves:

                if self.is_consistent(current_leaf.core, min_agreement_size, cut):
                    new_node_core = self.get_most_restrictive_core(current_leaf.core, cut)
                    new_node = TangleNode(cut_index + 1, core=new_node_core)

                    current_leaf.left = new_node
                    new_leaves.append(new_node)

                cut_c = 1 - cut
                if self.is_consistent(current_leaf.core, min_agreement_size, cut_c):
                    new_node_core = self.get_most_restrictive_core(current_leaf.core, cut_c)
                    new_node = TangleNode(-(cut_index + 1), core=new_node_core)
                    current_leaf.right = new_node
                    new_leaves.append(new_node)

            if len(new_leaves) == 0:
                break

            current_leaves = new_leaves

        self.cuts = self.cuts[:, :cut_index + 1]
        self.costs = self.costs[: cut_index + 1]

        if self.prune_depth is not None:
            self.root = prune_tree(self.root, self.prune_depth)

        self.root = self.condense_tree(self.root)
        self.contract_tree(self.root)

        return self.root

    def compute_tangles_reevaluate_cuts(self, data):
        # Chapter 5 in Thesis

        self.root = TangleNode(
            'root', core=np.empty((0, self.cuts.shape[0]), dtype=np.int8),
            is_root=True)

        self.root.cost_order = np.arange(self.cuts.shape[1])
        # Binary array to represent which cuts have been used in a branch
        # 0 represents unused, 1 represents used.
        self.root.used_cuts = np.zeros(self.cuts.shape[1], dtype=np.int8)
        self.root.costs = self.costs.copy()

        current_leaves = [self.root]

        if self.agreement_param < 1:
            min_agreement_size = self.agreement_param * self.cuts.shape[0]
        else:
            min_agreement_size = self.agreement_param

        for _ in range(self.cuts.shape[1]):

            new_leaves = []
            # Loop over all active leaves (leaf is active if it may still be possible to add child)
            for current_leaf in current_leaves:
                cut_index = int(current_leaf.cost_order[0])
                cut = self.cuts[:, cut_index]
                current_leaf.used_cuts[cut_index] = 1

                left, right = False, False

                if self.is_consistent(current_leaf.core, min_agreement_size, cut):
                    new_node_core = self.get_most_restrictive_core(current_leaf.core, cut)
                    new_node = TangleNode(cut_index + 1, core=new_node_core)
                    new_node.cost_order = current_leaf.cost_order[1:]
                    new_node.used_cuts = current_leaf.used_cuts.copy()
                    new_node.costs = current_leaf.costs.copy()
                    current_leaf.left = new_node
                    new_leaves.append(new_node)
                    left = True

                cut_c = 1 - cut
                if self.is_consistent(current_leaf.core, min_agreement_size, cut_c):
                    new_node_core = self.get_most_restrictive_core(
                        current_leaf.core, cut_c)
                    new_node = TangleNode(-(cut_index + 1), core=new_node_core)
                    new_node.cost_order = current_leaf.cost_order[1:]
                    new_node.used_cuts = current_leaf.used_cuts.copy()
                    new_node.costs = current_leaf.costs.copy()

                    current_leaf.right = new_node
                    new_leaves.append(new_node)
                    right = True

                if left and right:

                    def sub_costs(current_node, leaf):
                        # Recompute costs only on the data represented
                        # by the core Of the tangle
                        core_mask = leaf.get_core()

                        # In the rare case that a core of a tangle is empty.
                        # Yes.. that can happen.
                        if core_mask.sum() == 0:
                            return current_node.costs, current_node.cost_order

                        sub_data = data[core_mask == 1]
                        sub_cuts = self.cuts[core_mask == 1]
                        sub_cuts = sub_cuts[:, leaf.used_cuts == 0]
                        sub_data = MinMaxScaler().fit_transform(sub_data.copy())

                        costs = self.CostClass(sub_data).compute_(sub_cuts)
                        arg_sorted = np.argsort(costs)
                        # Bit of a hacky way so we can get cuts
                        # easily without recomputing used cuts.
                        # This computes the indices they would be in the original array
                        for idx in np.where(leaf.used_cuts == 1)[0]:
                            arg_sorted[arg_sorted >= idx] += 1

                        return costs, arg_sorted

                    left_leaf = new_leaves[-2]
                    costs, arg_sorted = sub_costs(current_leaf, left_leaf)
                    new_leaves[-2].cost_order = arg_sorted
                    new_leaves[-2].costs = costs
                    current_leaf.left_costs = costs
                    # current_leaf.left_cost_order = arg_sorted

                    right_leaf = new_leaves[-1]
                    costs, arg_sorted = sub_costs(current_leaf, right_leaf)
                    new_leaves[-1].cost_order = arg_sorted
                    new_leaves[-1].costs = costs
                    current_leaf.right_costs = costs
                    # new_leaves[-1].cost_order = sub_costs(
                    #     current_leaf, right_leaf)

            current_leaves = new_leaves
            if len(current_leaves) == 0:
                break

        self.root = self.condense_tree(self.root)
        self.contract_tree(self.root)

        return self.root

    def get_cuts_from_names(self, names):
        names = np.array(list(names))
        complements = names < 0

        cut_indices = np.abs(names) - 1

        relevant_cuts = self.cuts[:, cut_indices]

        relevant_cuts[:, complements] = 1 - relevant_cuts[:, complements]
        return relevant_cuts, cut_indices

    def compute_soft_predictions(self, node):
        if node is None:
            return

        if node is not None and node.is_root:

            node.p = np.ones(self.cuts.shape[0])

            if self.prune_from_projection:
                node.data_idx = np.full(self.cuts.shape[0], True)

        if node.left is not None and node.right is not None:

            cuts, cut_indices = self.get_cuts_from_names(
                node.distinguished_cuts)

            if self.recompute_costs_on_split:
                # Case where final bipartition is a split
                # Then left_and_right_costs are empty
                local_weights_left = np.array([1])
                local_weights_right = np.array([1])
                if node.left_costs.size == 0:
                    local_weights_left = np.array([1])

                if node.right_costs.size == 0:
                    local_weights_right = np.array([1])

                # Used cuts can be used as a mask to fill back into their
                # original positions, so we can retrieve the correct weights
                # from the indices of the distinguished cuts the cost of the split
                # isn't included in the weights, but it's conveniently 1
                # from used_cuts  Which is the highest weight.
                # Probably not the most elegant solution but it works.. :-)
                weights_left = node.used_cuts.copy().astype(np.float64)
                weights_left[weights_left == 0] = local_weights_left
                weights_left = weights_left[cut_indices]

                weights_right = node.used_cuts.copy().astype(np.float64)
                weights_right[weights_right == 0] = local_weights_right
                weights_right = weights_right[cut_indices]

                p_left = np.sum(cuts * weights_left, axis=1) / np.sum(weights_left)
                p_right = np.sum((1 - cuts) * weights_right, axis=1) / np.sum(weights_right)

                total = p_left + p_right
                p_left = p_left / total
                p_right = p_right / total

            else:
                weights = self.weights[cut_indices]
                p_left = np.sum(cuts * weights, axis=1) / np.sum(weights)
                p_right = 1 - p_left

            node.left.p = p_left * node.p
            node.right.p = p_right * node.p

            # Can we do this while constructing the tangles somehow for speed-boost?
            # Perhaps on the tangle cores?
            # Chapter 6 in thesis.
            if self.prune_from_projection:
                node.left.data_idx = node.left.p > (node.p / 2)
                node.right.data_idx = node.right.p > (node.p / 2)

                msk = node.data_idx

                mean_left = np.average(
                    self.data[msk], axis=0, weights=node.left.p[msk])

                mean_right = np.average(
                    self.data[msk], axis=0, weights=node.right.p[msk])

                direction_vector = mean_right - mean_left

                def project_onto_line(data, line_direction, line_origin):
                    # Normalize the direction vector
                    line_direction_norm = line_direction / np.linalg.norm(line_direction)
                    # Compute the projection of each point onto the line
                    projections = np.dot(data - line_origin, line_direction_norm)
                    return projections

                projected_data = project_onto_line(
                    self.data[msk], direction_vector, mean_left)

                hist, _ = np.histogram(
                    projected_data, bins=16, density=True)

                peaks, _ = find_peaks(hist)

                if self.plot_hist:
                    plotting.plot_histogram_projection(
                        node, msk, self.data,
                        mean_left, mean_right, direction_vector, hist, peaks
                    )

                if len(peaks) == 1:
                    node.left = None
                    node.right = None

            self.compute_soft_predictions(node.left)
            self.compute_soft_predictions(node.right)

    def fit(self, data):

        if self.cuts is None:
            self.cuts = self.cut_function(data)

        if self.costs is None:
            cost_function = self.CostClass(data)
            self.cuts, self.costs = cost_function.compute(self.cuts)

        if not np.all(self.costs[:-1] <= self.costs[1:]):
            raise ValueError("Costs must be sorted in increasing order")

        if not np.all(np.isin(self.cuts, [0, 1])):
            raise ValueError("Cuts must be binary")

        if self.recompute_costs_on_split:
            self.compute_tangles_reevaluate_cuts(data)
        else:
            self.compute_tangles()

    def predict(self):
        self.weights = np.exp(-normalize(self.costs))

        self.compute_soft_predictions(self.root)
        self.soft_predictions = np.column_stack([
            node.p for node in self.root.get_leaves()])
        self.hard_predictions = np.argmax(self.soft_predictions, axis=1)
        return self.hard_predictions

    def fit_predict(self, data):
        self.data = data
        self.fit(data)
        return self.predict()

    def plot_tree(self):
        if not self.contracted:
            raise ValueError('Tree must be contracted before plotting')
        self.root.get_leaves(print_=True)

    def plot_tangles(self, data, node=None, title='Tangle Cores'):
        if node is None:
            node = self.root

        leaves = node.get_leaves()
        clusters = np.array([np.bitwise_and.reduce(leaf.core, axis=0)
                             for leaf in leaves])

        for i in range(len(leaves)):
            leaves[i].value = f'T{i + 1} {leaves[i].value}'

        if np.any(clusters.sum(axis=0) > 1):
            raise ValueError('Tangles are not disjoint')

        tangle_labels = np.argmax(clusters, axis=0) + 1
        tangle_labels[np.sum(clusters, axis=0) == 0] = 0
        unique_labels = np.unique(tangle_labels)

        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))

        for i, label in enumerate(unique_labels):
            cluster = data[tangle_labels == label]
            if label == 0:
                plt.scatter(cluster[:, 0], cluster[:, 1], color='brown',
                            s=30, alpha=0.5)
            else:
                plt.scatter(cluster[:, 0], cluster[:, 1], color=colors[i],
                            label=f"Tangle Core {label}", s=30, cmap='viridis', alpha=0.5
                            )

        plt.title(title)
        plt.legend()
        plt.grid(True)
