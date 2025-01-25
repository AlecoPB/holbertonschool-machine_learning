#!/usr/bin/env python3
"""
This is some documentation
"""
import numpy as np
Isolation_Random_Tree = __import__('10-isolation_tree').Isolation_Random_Tree

class Isolation_Forest:
    """
    Implementation of the Isolation Forest for detecting outliers.
    """
    def __init__(self, n_trees=100, max_depth=10, seed=0):
        """
        Initialize the Isolation Forest.

        Args:
            n_trees (int): Number of trees in the forest.
            max_depth (int): Maximum depth of each tree.
            seed (int): Random seed for reproducibility.
        """
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.seed = seed
        self.trees = []

    def fit(self, explanatory):
        """
        Train the Isolation Forest on the dataset by creating multiple
        Isolation Random Trees.

        Args:
            explanatory (np.ndarray): The dataset to train the forest on.
        """
        self.explanatory = explanatory
        self.trees = []

        for i in range(self.n_trees):
            tree = Isolation_Random_Tree(
                max_depth=self.max_depth,
                seed=self.seed + i
            )
            tree.fit(explanatory)
            self.trees.append(tree)

    def average_depths(self, explanatory):
        """
        Compute the average depths of individuals in the forest.

        Args:
            explanatory (np.ndarray): The dataset to compute depths for.

        Returns:
            np.ndarray: An array of average depths for each individual.
        """
        depths = np.zeros((len(self.trees), explanatory.shape[0]))

        for i, tree in enumerate(self.trees):
            depths[i] = tree.predict(explanatory)

        return np.mean(depths, axis=0)

    def suspects(self, explanatory, n_suspects):
        """
        Identify the top n_suspects likely to be outliers based on
        minimum average depths.

        Args:
            explanatory (np.ndarray): The dataset to evaluate.
            n_suspects (int): Number of top suspects to return.

        Returns:
            np.ndarray: Indices of the n_suspects most likely to be outliers.
        """
        # Compute average depths
        avg_depths = self.average_depths(explanatory)

        # Sort individuals by increasing average depth (outliers have low depth)
        sorted_indices = np.argsort(avg_depths)

        # Return the top n_suspects indices
        return sorted_indices[:n_suspects]
