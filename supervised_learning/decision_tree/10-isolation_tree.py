#!/usr/bin/env python3
"""Define class `Isolation_Random_Tree`."""
import numpy as np

Node = __import__('8-build_decision_tree').Node
Leaf = __import__('8-build_decision_tree').Leaf


class Isolation_Random_Tree:
    """Define isolation tree with random split criterion."""

    def __init__(self, max_depth=10, seed=0, root=None):
        """Initialize isolation tree."""
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = np.array([])
        self.max_depth = max_depth
        self.predict = None
        self.min_pop = 1
        self.split_criterion = lambda x: 0

    def __str__(self):
        """User-friendly representation of `Isolation_Random_Tree`."""
        return self.root.__str__() + "\n"

    def depth(self):
        """Calculate depth of the isolation random tree."""
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """Count the number of nodes in the decision tree."""
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def update_bounds(self):
        """Update the bounds of the current data for each feature."""
        self.root.update_bounds_below()

    def get_leaves(self):
        """List all leaves of the tree."""
        return self.root.get_leaves_below()

    def update_predict(self):
        """Compute prediction function."""
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()

        def indicators(A):
            """Calculate indicators array of `n_leaves * n_elements`."""
            return np.array([leaf.indicator(A) for leaf in leaves])
        # indicators is an array of shape = (n_leaves, n_elements_indicators)
        values = np.array([leaf.value for leaf in leaves])
        self.predict = lambda A: values @ indicators(A)

    def np_extrema(self, arr):
        """Calculate extrema of array."""
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        """Define a random split criterion."""
        diff = 0
        feature_min = -1 * np.inf
        feature_max = np.inf
        feature = 0
        while diff == 0:
            # select a feature at random
            feature = self.rng.integers(0, self.explanatory.shape[1])
            # calculate extrema of the explanatory (training data) under a
            # sub_population mask (only elements under the current node taken
            # into account
            feature_min, feature_max = self.np_extrema(
                    self.explanatory[:, feature][node.sub_population])
            diff = feature_max - feature_min
        x = self.rng.uniform()
        # select threshold at random between the extrema of the (subset of)
        # explanatory data
        threshold = (1-x) * feature_min + x * feature_max
        return feature, threshold

    def get_leaf_child(self, node, sub_population):
        """Create leaf node with no value."""
        leaf_child = Leaf(node.depth + 1)
        leaf_child.depth = node.depth + 1
        leaf_child.subpopulation = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        """Create node with its corresponding sub_population."""
        n = Node()
        n.depth = node.depth + 1
        n.sub_population = sub_population
        return n

    def fit_node(self, node):
        """Create child nodes with features and thresholds chosen at random."""
        node.feature, node.threshold = self.random_split_criterion(node)

        feature_values = self.explanatory[:, node.feature]
        left_population = (
            node.sub_population & (feature_values > node.threshold)
        )
        right_population = (
            node.sub_population & (feature_values <= node.threshold)
            )
        child_depth = node.depth + 1
        # Is left node a leaf ?
        is_left_leaf = (
            np.sum(left_population) <= self.min_pop
            or child_depth == self.max_depth
            )

        if is_left_leaf:
            node.left_child = self.get_leaf_child(node, left_population)
        else:
            node.left_child = self.get_node_child(node, left_population)
            self.fit_node(node.left_child)

        # Is right node a leaf ?
        is_right_leaf = (
            np.sum(right_population) <= self.min_pop
            or child_depth == self.max_depth
        )

        if is_right_leaf:
            node.right_child = self.get_leaf_child(node, right_population)
        else:
            node.right_child = self.get_node_child(node, right_population)
            self.fit_node(node.right_child)

    def fit(self, explanatory, verbose=0):
        """Fit tree."""
        self.split_criterion = self.random_split_criterion
        self.explanatory = explanatory
        self.root.sub_population = np.ones(explanatory.shape[0],
                                           dtype='bool')

        self.fit_node(self.root)
        self.update_predict()

        if verbose == 1:
            print(f"""  Training finished.
    - Depth                     : {self.depth()}
    - Number of nodes           : {self.count_nodes()}
    - Number of leaves          : {self.count_nodes(only_leaves=True)}""")
