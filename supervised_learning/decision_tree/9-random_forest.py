import numpy as np
from scipy.stats import mode
Decision_Tree = __import__('8-build_decision_tree').Decision_Tree

class Random_Forest():
    def __init__(self, n_trees=100, max_depth=10, min_pop=1, seed=0):
        self.numpy_predicts = []  # List of individual tree prediction functions
        self.target = None  # Training target
        self.numpy_preds = None  # Placeholder for predictions
        self.n_trees = n_trees  # Number of trees in the forest
        self.max_depth = max_depth  # Maximum depth for each tree
        self.min_pop = min_pop  # Minimum population per node
        self.seed = seed  # Random seed for reproducibility

    def predict(self, explanatory):
        """
        Predict the class for each individual in the explanatory dataset
        by taking the mode of the predictions from all trees.
        """
        # Collect predictions from all trees
        predictions = np.array([tree(explanatory) for tree in self.numpy_preds])  # Shape: (n_trees, n_individuals)

        # Calculate the mode (most frequent prediction) along the tree axis
        final_predictions, _ = mode(predictions, axis=0)
        return final_predictions.flatten()

    def fit(self, explanatory, target, n_trees=100, verbose=0):
        """
        Train the random forest by fitting `n_trees` decision trees on the training data.
        """
        self.target = target
        self.explanatory = explanatory
        self.numpy_preds = []  # Store prediction functions for each tree
        depths = [] 
        nodes = [] 
        leaves = []
        accuracies = []

        for i in range(n_trees):
            # Train a new decision tree
            T = Decision_Tree(max_depth=self.max_depth, min_pop=self.min_pop, seed=self.seed + i)
            T.fit(explanatory, target)
            
            # Store the tree's prediction function
            self.numpy_preds.append(T.predict)
            
            # Collect metrics for verbose output
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
    - Accuracy of the forest on td   : {self.accuracy(self.explanatory, self.target)}""")

    def accuracy(self, test_explanatory, test_target):
        """
        Compute the accuracy of the random forest on the test dataset.
        """
        return np.sum(np.equal(self.predict(test_explanatory), test_target)) / test_target.size
