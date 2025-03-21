#!/usr/bin/env python3
"""
This is some documentation
"""
import numpy as np
Isolation_Random_Tree = __import__('10-isolation_tree').Isolation_Random_Tree


class Isolation_Random_Forest:
    """
    Implementation of the Isolation Forest for detecting outliers.
    """
    def __init__(self, n_trees=100, max_depth=10, seed=0):
        self.numpy_predicts = []
        self.target = None
        self.numpy_preds = None
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.seed = seed

    def predict(self, explanatory):
        """
        Average prediciton for each tree
        """
        predictions = np.array([f(explanatory) for f in self.numpy_preds])
        return predictions.mean(axis=0)

    def fit(self, explanatory, n_trees=100, verbose=0):
        """
        Train the Isolation Forest on the dataset by creating multiple
        Isolation Random Trees.

        Args:
            explanatory (np.ndarray): The dataset to train the forest on.
        """
        self.explanatory = explanatory
        self.numpy_preds = []
        depths = []
        nodes = []
        leaves = []
        for i in range(n_trees):
            T = Isolation_Random_Tree(
                max_depth=self.max_depth, seed=self.seed+i)
            T.fit(explanatory)
            self.numpy_preds.append(T.predict)
            depths.append(T.depth())
            nodes.append(T.count_nodes())
            leaves.append(T.count_nodes(only_leaves=True))
        if verbose == 1:
            print(f"""  Training finished.
    - Mean depth                     : { np.array(depths).mean()      }
    - Mean number of nodes           : { np.array(nodes).mean()       }
    - Mean number of leaves          : { np.array(leaves).mean()      }""")

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
        depths = self.predict(explanatory)
        sorted_indices = np.argsort(depths)

        # Using these indices to get the corresponding suspect rows in
        # explanatory (the dataset) and their depths (the predictions)
        return explanatory[sorted_indices[:n_suspects]], \
            depths[sorted_indices[:n_suspects]]
