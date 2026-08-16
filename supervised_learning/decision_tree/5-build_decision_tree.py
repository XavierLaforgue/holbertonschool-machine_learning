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
        self.sub_population = None
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
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

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
