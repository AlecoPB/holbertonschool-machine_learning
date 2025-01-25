#!/usr/bin/env python3
"""
This is some documentation
"""
import numpy as np
Node = __import__('8-build_decision_tree').Node
Leaf = __import__('8-build_decision_tree').Leaf


class Isolation_Random_Tree:
    """
    Define and initialize an isolation tree
    """
    def __init__(self, max_depth=10, min_pop=1, seed=0):
        """
        Initialize the isolation tree with constraints on depth and population.
        """
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.seed = seed
        self.root = None
        np.random.seed(self.seed)

    def fit(self, explanatory):
        """
        Train the isolation tree by recursively splitting the data until
        `max_depth` or `min_pop` is reached.
        """
        self.explanatory = explanatory
        self.root = Node()
        self.root.sub_population = np.arange(explanatory.shape[0])  # Initialize with all indices
        self._fit_node(self.root)

    def _fit_node(self, node):
        """
        Recursively fit a node by selecting a random feature and threshold for splitting.
        """
        # Check stopping conditions
        if len(node.sub_population) <= self.min_pop or node.depth >= self.max_depth:
            node.is_leaf = True
            return

        # Select a random feature and threshold
        n_features = self.explanatory.shape[1]
        feature = np.random.randint(n_features)
        feature_values = self.explanatory[node.sub_population, feature]
        threshold = np.random.uniform(feature_values.min(), feature_values.max())
        
        node.feature = feature
        node.threshold = threshold

        # Partition the sub_population
        left_indices = node.sub_population[feature_values > threshold]
        right_indices = node.sub_population[feature_values <= threshold]

        # Create left child
        if len(left_indices) > 0:
            node.left_child = Node()
            node.left_child.depth = node.depth + 1
            node.left_child.sub_population = left_indices
            self._fit_node(node.left_child)

        # Create right child
        if len(right_indices) > 0:
            node.right_child = Node()
            node.right_child.depth = node.depth + 1
            node.right_child.sub_population = right_indices
            self._fit_node(node.right_child)

    def predict(self, explanatory):
        """
        Predict the depth of the leaf where each individual falls.
        """
        depths = np.zeros(explanatory.shape[0])

        for i, individual in enumerate(explanatory):
            depths[i] = self._predict_node(self.root, individual)

        return depths

    def _predict_node(self, node, individual):
        """
        Recursively traverse the tree to find the depth of the leaf for a single individual.
        """
        if node.is_leaf or node.left_child is None or node.right_child is None:
            return node.depth

        if individual[node.feature] > node.threshold:
            return self._predict_node(node.left_child, individual)
        else:
            return self._predict_node(node.right_child, individual)
