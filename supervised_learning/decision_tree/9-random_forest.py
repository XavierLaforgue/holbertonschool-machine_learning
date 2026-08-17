#!/usr/bin/env python3
"""Define the class Random_Forest."""
import numpy as np
Decision_Tree = __import__('8-build_decision_tree').Decision_Tree


class Random_Forest():
    """Define a random forest."""

    def __init__(self, n_trees=100, max_depth=10, min_pop=1, seed=0):
        """Initialize an instance of random forest."""
        self.numpy_predicts = []
        self.target = None
        self.numpy_preds = []
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.seed = seed
        self.explanatory = np.array([])

    def predict(self, explanatory):
        """Predict the value of the leaf the individuals will reach."""
        # Initialize an empty list to store predictions from individual trees
        tree_predictions = []

        # Generate predictions for each tree in the forest
        for preds in self.numpy_preds:
            tree_predictions.append(preds(explanatory))
        preds_arr = np.array(tree_predictions)  # array (n_trees, n_elements)
        # Calculate the mode (most frequent) prediction for each example

        most_freq_preds = []
        # explanatory is (n_elements, n_features)
        # preds_arr is (n_trees, n_elements)
        for elem in range(preds_arr.shape[1]):
            values, counts = np.unique(preds_arr[:, elem], return_counts=True)
            most_freq_preds.append(values[np.argmax(counts)])
        return np.array(most_freq_preds)

    def fit(self, explanatory, target, n_trees=None, verbose=0):
        """Fit many decision trees and print descriptive results."""
        if n_trees is None:
            n_trees = self.n_trees
        self.target = target
        self.explanatory = explanatory
        self.numpy_preds = []
        depths = []
        nodes = []
        leaves = []
        accuracies = []
        for i in range(n_trees):
            T = Decision_Tree(
                max_depth=self.max_depth, min_pop=self.min_pop,
                seed=self.seed + i
            )
            T.fit(explanatory, target)
            self.numpy_preds.append(T.predict)  # T.predict is a callable
            # expecting a sample (e.g., explanatory)
            depths.append(T.depth())
            nodes.append(T.count_nodes())
            leaves.append(T.count_nodes(only_leaves=True))
            accuracies.append(T.accuracy(T.explanatory, T.target))
        if verbose == 1:
            print(f"""  Training finished.
    - Mean depth                     : {np.array(depths).mean()}
    - Mean number of nodes           : {np.array(nodes).mean()}
    - Mean number of leaves          : {np.array(leaves).mean()}
    - Mean accuracy on training data : {np.array(accuracies).mean()}
    - Accuracy of the forest on td   : {self.accuracy(self.explanatory,
                  self.target)}""")

    def accuracy(self, test_explanatory, test_target):
        """Calculate prediction accuracy averaged over sample size."""
        return (np.sum(np.equal(self.predict(test_explanatory), test_target))
                / test_target.size)
