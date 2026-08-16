#!/usr/bin/env python3
"""Define the `Decision_tree`, `Node`, and `Leaf` classes."""
import numpy as np


class Node:
    """Define a node in a decision tree."""

    def __init__(self,
                 feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        """Initialize instance of a `Node` of the `Decision_Tree`."""
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = np.array([])
        self.depth = depth
        self.upper = {}
        self.lower = {}
        self.indicator = None

    def left_child_add_prefix(self, text):
        """Add style prefix for a left-child node."""
        lines = text.split("\n")
        new_text = "    +--" + lines[0]
        for x in lines[1:]:
            new_text += "\n" + ("    |  " + x)
        return (new_text)

    def right_child_add_prefix(self, text):
        """Add style prefix for a right-child node."""
        lines = text.split("\n")
        new_text = "    +--" + lines[0]
        for x in lines[1:]:
            new_text += "\n" + ("       " + x)
        return (new_text)

    def __str__(self):
        """Return user-friendly string representation of `Node`."""
        if self.is_root:
            node_type = "root "
        else:
            node_type = "-> node "
        description = f"[feature={self.feature}, threshold={self.threshold}]"
        sub_tree = node_type + description
        if self.left_child:
            sub_tree += "\n"\
                    + self.left_child_add_prefix(self.left_child.__str__())
        if self.right_child:
            sub_tree += "\n"\
                    + self.right_child_add_prefix(self.right_child.__str__())
        return sub_tree

    def max_depth_below(self):
        """Calculate maximum depth of nodes below the current instance."""
        if self.is_leaf:
            return self.depth
        if self.left_child and self.right_child:
            return max(self.left_child.max_depth_below(),
                       self.right_child.max_depth_below())
        return (self.left_child.max_depth_below() if self.left_child
                else self.right_child.max_depth_below() if self.right_child
                else self.depth)

    def count_nodes_below(self, only_leaves=False):
        """Count number of nodes in subtree with current node as root."""
        if self.is_leaf:
            print("Should be instance of `Leaf`, but it is instance of Node")
            return 1
        left = self.left_child
        right = self.right_child
        count = 1 if not only_leaves else 0
        if left:
            count += left.count_nodes_below(only_leaves=only_leaves)
        if right:
            count += right.count_nodes_below(only_leaves=only_leaves)
        return count

    def get_leaves_below(self):
        """List all leaves of below current node."""
        leaves = []
        if self.left_child:
            leaves.extend(self.left_child.get_leaves_below())
        if self.right_child:
            leaves.extend(self.right_child.get_leaves_below())
        return leaves

    def update_bounds_below(self):
        """Update the known bounds of the current data for each feature."""
        if self.is_root:
            self.upper = {0: np.inf}
            self.lower = {0: -1*np.inf}
        assert self.left_child is not None and self.right_child is not None
        assert self.feature is not None and self.threshold is not None
        for child in [self.left_child, self.right_child]:
            child.upper = dict(self.upper)
            child.lower = dict(self.lower)
        self.left_child.lower[self.feature] = self.threshold
        self.right_child.upper[self.feature] = self.threshold
        for child in [self.left_child, self.right_child]:
            child.update_bounds_below()

    def update_indicator(self):
        """Update indicator of compliance with the node condition."""
        def is_large_enough(A: np.ndarray):
            """Indicate if all features > lower bounds for each element.

            Return a 1D numpy array of size `n_individuals` so that the `i`-th
            element of the later is `True` if the `i`-th individual has all its
            features > the lower bounds.
            """
            return np.all(
                    np.array([np.greater(A[:, feature], self.lower[feature])
                              for feature in self.lower]),
                    axis=0)

        def is_small_enough(A: np.ndarray):
            """Indicate if all features <= upper bounds for each element.

            Return a 1D numpy array of size `n_individuals` so that the `i`-th
            element of the later is `True` if the `i`-th individual has all its
            features <= the upper bounds.
            """
            return np.all(
                    np.array([np.less_equal(A[:, feature], self.upper[feature])
                              for feature in self.upper]),
                    axis=0)
        self.indicator = lambda A: np.all(
                np.array([is_large_enough(A), is_small_enough(A)]),
                axis=0)
        # self.indicator answers the question: which elements/samples end up in
        # this leaf

    def pred(self, x):
        """Compute prediction wih inefficient conditionals."""
        assert self.left_child is not None
        assert self.right_child is not None
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        return self.right_child.pred(x)


class Leaf(Node):
    """Define a leaf node in a decision tree."""

    def __init__(self, value, depth=None):
        """Initialize `Leaf` instance."""
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def __str__(self):
        """Return user-friendly string representation of `Leaf`."""
        return (f"-> leaf [value={self.value}]")

    def max_depth_below(self):
        """Return leaf's depth."""
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """Return 1 (leaf = 1 node + 0 children)."""
        return 1

    def get_leaves_below(self):
        """List itself."""
        return [self]

    def update_bounds_below(self):
        """Not applicable."""

    def pred(self, x):
        """Return value of leaf."""
        return self.value


class Decision_Tree:
    """Define a decision tree."""

    def __init__(self,
                 max_depth=10, min_pop=1, seed=0, split_criterion="random",
                 root=None):
        """Initialize `Decision_Tree` instance."""
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = np.array([])
        self.target = np.array([])
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion_type = split_criterion
        self.split_criterion = lambda x: x
        self.predict = lambda x: x

    def __str__(self):
        """Return user-friendly string representation of `Decision_Tree`."""
        return self.root.__str__() + "\n"

    def depth(self):
        """Calculate depth the decision tree."""
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """Count the number of nodes in the decision tree."""
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def get_leaves(self):
        """List all leaves of the tree."""
        return self.root.get_leaves_below()

    def update_bounds(self):
        """Update the bounds of the current data for each feature."""
        self.root.update_bounds_below()

    def pred(self, x):
        """Compute prediction with inefficient loops and conditionals."""
        return self.root.pred(x)

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

    def fit(self, explanatory, target, verbose=0):
        """Fit tree."""
        # Select split criterion: a method that takes a node and calculates its
        # split.
        if self.split_criterion_type == "random":
            self.split_criterion = self.random_split_criterion
        else:
            self.split_criterion = self.Gini_split_criterion
        # store training 2D data of shape (n_elements, n_features)
        self.explanatory = explanatory
        # store target 1D data of shape (n_elements)
        self.target = target
        # set sub_population as 1D array of elements/individuals that visit
        # the node in question: an array of size n_elements for the root node
        self.root.sub_population = np.ones_like(self.target, dtype=bool)

        self.fit_node(self.root)

        self.update_predict()
        if verbose == 1:
            print(f"""  Training finished.
    - Depth                     : {self.depth()}
    - Number of nodes           : {self.count_nodes()}
    - Number of leaves          : {self.count_nodes(only_leaves=True)}
    - Accuracy on training data : {self.accuracy(self.explanatory, self.target)}"""
                  )

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

    def Gini_split_criterion(self, node):
        """Calculate the Gini split criterion."""
        return node.sub_populatin[0], node.sub_population[0]

    def fit_node(self, node):
        """Create child nodes with features and thresholds as per fitting."""
        # determine feature and threshold for the node given a split criterion
        # (random by default)
        node.feature, node.threshold = self.split_criterion(node)

        # use subset of explanatory values for the feature of the node
        feature_values = self.explanatory[:, node.feature]
        # the subpopulation to the left is the subpopulation for which the
        # values of the features in the explanatory dataset is stictly greater
        # than the selected threshold
        left_population = (
                node.sub_population & (feature_values > node.threshold)
                )
        # the subpopulation to the right is the subpopulation for which the
        # values of the features in the explanatory datase is less or equal to
        # the selected threshold
        right_population = (
                node.sub_population & (feature_values <= node.threshold)
                )

        # the left child node is a leaf if either:
        # - it contains less than min_pop individuals, or
        # - its depth equals max_depth, or
        # - all the individuals of the training set have the same target value
        child_depth = node.depth + 1
        target_left = self.target[left_population]
        is_left_leaf = (
                np.sum(left_population) < self.min_pop
                or child_depth == self.max_depth
                or np.all(target_left == target_left[0])
                )

        # If the left child is a leaf the we create the leaf object and set it
        # as left child of the current node.
        # Else, we get the child node and recursively apply fit_node on it till
        # we find the leaf.
        if is_left_leaf:
            node.left_child = self.get_leaf_child(node, left_population)
        else:
            node.left_child = self.get_node_child(node, left_population)
            self.fit_node(node.left_child)

        # same rules as for the left child node
        target_right = self.target[right_population]
        is_right_leaf = (
                np.sum(right_population) < self.min_pop
                or child_depth == self.max_depth
                or np.all(target_right == target_right[0])
                )

        # create and assign leaf if right child is leaf, otherwise get the
        # child node and pass it to fit_node.
        if is_right_leaf:
            node.right_child = self.get_leaf_child(node, right_population)
        else:
            node.right_child = self.get_node_child(node, right_population)
            self.fit_node(node.right_child)

    # the value of the leaf is the most common value in the sub_population
    def get_leaf_child(self, node, sub_population):
        """Create leaf node with most frequent value in sub_population."""
        value, counts = np.unique(self.target[sub_population],
                                  return_counts=True)
        # np.argmax(arr) returns index of max value in arr
        most_represented = value[np.argmax(counts)]
        leaf_child = Leaf(most_represented)
        leaf_child.depth = node.depth + 1
        leaf_child.sub_population = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        """Create node with its corresponding sub_population."""
        n = Node()
        n.depth = node.depth + 1
        n.sub_population = sub_population
        return n

    def accuracy(self, test_explanatory, test_target):
        """Calculate accuracy of fit."""
        return np.sum(
                np.equal(self.predict(test_explanatory), test_target)
                ) / test_target.size
