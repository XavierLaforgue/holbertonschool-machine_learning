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

    def max_depth_below(self):
        """Calculate maximum depth of nodes below the current instance."""
        if self.is_leaf:
            return self.depth
        return max(self.max_depth_below(self.left_child),
                   self.max_depth_below(self.right_child))


class Leaf(Node):
    """Define a leaf node in a decision tree."""

    def __init__(self, value, depth=None):
        """Initialize `Leaf` instance."""
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """Return leaf's depth."""
        return self.depth


class Decision_Tree():
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

    def depth(self):
        """Calculate depth the decision tree."""
        return self.root.max_depth_below()
