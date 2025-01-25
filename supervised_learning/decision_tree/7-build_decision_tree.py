#!/usr/bin/env python3
"""
This is some documentation
"""
import numpy as np


class Node:
    """
    A node for a tree
    """
    def __init__(self, feature=None,
                 threshold=None,
                 left_child=None,
                 right_child=None,
                 is_root=False,
                 depth=0):
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        """
        Max depth
        """
        if self.is_leaf:
            return self.depth

        return max(self.left_child.max_depth_below(),
                   self.right_child.max_depth_below())

    def count_nodes_below(self, only_leaves=False):
        """
        We're counting the nodes present
        We have the option to just count the leaves
        """
        if only_leaves and self.is_leaf:
            return 1

        if not self.is_leaf:
            # NOTE Counting the current node only if only_leaves == False
            return self.left_child.count_nodes_below(only_leaves=only_leaves)\
                + self.right_child.count_nodes_below(only_leaves=only_leaves)\
                + (not only_leaves)

    def __str__(self):
        """
        Prints string representation of the node and its children.
        """

        if self.is_root:
            s = "root"
        else:
            s = "-> node"

        return f"{s} [feature={self.feature}, threshold={self.threshold}]\n"\
            + self.left_child_add_prefix(str(self.left_child))\
            + self.right_child_add_prefix(str(self.right_child))

    def left_child_add_prefix(self, text):
        """
        Adds the string representation of the left child to the given text
        """
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("    |  " + x) + "\n"
        return (new_text)

    def right_child_add_prefix(self, text):
        """
        Adds the string representation of the right child to the given text
        """
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("       " + x) + "\n"

        return (new_text.rstrip())

    def get_leaves_below(self):
        """
        Returns the list of all leaves of the tree.
        """
        return self.left_child.get_leaves_below()\
            + self.right_child.get_leaves_below()

    def update_bounds_below(self):
        """
        Recursively compute, for each node, two dictionaries
        stored as attributes Node.lower and Node.upper.
        """
        if self.is_root:
            self.lower, self.upper = {0: -np.inf}, {0: np.inf}

        for child in [self.left_child, self.right_child]:
            child.lower, child.upper = self.lower.copy(), self.upper.copy()

        self.left_child.lower[self.feature] = max(
            self.threshold,
            self.left_child.lower.get(self.feature, self.threshold)
        )
        self.right_child.upper[self.feature] = min(
            self.threshold,
            self.right_child.upper.get(self.feature, self.threshold)
        )

        self.left_child.update_bounds_below()
        self.right_child.update_bounds_below()

    def update_indicator(self):
        """
        Compute the indicator function from the Node.lower
        and Node.upper dictionaries and store it in the
        Node.indicator attribute.
        """

        def is_large_enough(x):
            """
            Returns a 1D numpy array of size n_individuals,
            where the i-th element is True if the i-th individual's
            features are >= the corresponding lower bounds.
            """
            return np.all([x[:, feature] >= threshold for feature,
                           threshold in self.lower.items()], axis=0)

        def is_small_enough(x):
            """
            Returns a 1D numpy array of size n_individuals,
            where the i-th element is True if the i-th individual's
            features are <= the corresponding upper bounds.
            """
            return np.all([x[:, feature] <= threshold for feature,
                           threshold in self.upper.items()], axis=0)

        # The indicator is True for individuals who satisfy both conditions
        self.indicator = lambda x: np.logical_and(is_large_enough(x),
                                                  is_small_enough(x))

    def pred(self, x):
        """
        Predicts a label
        """
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        else:
            return self.right_child.pred(x)


class Leaf(Node):
    """
    A leaf on a tree
    """
    def __init__(self, value, depth=None):
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """
        Depth of leaf
        """
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """
        Count nodes for leaves
        """
        return 1

    def __str__(self):
        return (f"-> leaf [value={self.value}]")

    def get_leaves_below(self):
        """
        Helps return the list of all leaves of the tree.
        """
        return [self]

    def update_bounds_below(self):
        """
        Passes
        """
        pass

    def pred(self, x):
        """
        Returns self
        """
        return self.value


class Decision_Tree():
    """
    A tree
    """
    def __init__(self, max_depth=10,
                 min_pop=1,
                 seed=0,
                 split_criterion="random",
                 root=None):
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
        """
        Total depth of tree
        """
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """
        Count nodes for nodes
        """
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def __str__(self):
        return f"{self.root.__str__()}\n"

    def get_leaves(self):
        """
        Returns the list of all the leaves
        """
        return self.root.get_leaves_below()

    def update_bounds(self):
        """
        Updates bounds
        """
        self.root.update_bounds_below()

    def pred(self, x):
        """
        Starts prediction from root
        """
        return self.root.pred(x)

    def update_predict(self):
        """
        Computes the prediction function
        """
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()

        self.predict = lambda A: np.array([self.root.pred(x) for x in A])

    def fit(self, explanatory, target, verbose=0) :
        if self.split_criterion == "random" : 
                self.split_criterion = self.random_split_criterion
        else : 
                self.split_criterion = self.Gini_split_criterion
        self.explanatory = explanatory
        self.target      = target
        self.root.sub_population = np.ones_like(self.target,dtype='bool')

        self.fit_node(self.root)

        self.update_predict()

        if verbose==1 :
            print(f"""  Training finished.
    - Depth                     : { self.depth()}
    - Number of nodes           : { self.count_nodes()}
    - Number of leaves          : { self.count_nodes(only_leaves=True)}
    - Accuracy on training data : { self.accuracy(self.explanatory,self.target)}""")

    def np_extrema(self, arr):
        """
        Compute min and max values
        """
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        """
        Selects a feature to use as the criteraia (random)
        """
        diff = 0
        while diff == 0:
            feature = self.rng.integers(0, self.explanatory.shape[1])
            feature_min, feature_max = self.np_extrema(
                self.explanatory[:, feature][node.sub_population])
            diff = feature_max - feature_min
        x = self.rng.uniform()
        threshold = (1 - x) * feature_min + x * feature_max
        return feature, threshold

    def fit_node(self, node):
        """
        Fit a single node of the decision tree. Split the population of the node
        into left and right subpopulations based on the splitting criterion.
        """
        # Determine the feature and threshold for splitting
        node.feature, node.threshold = self.split_criterion(node)

        # Create a boolean mask for the maximum criterion
        max_criterion = self.explanatory[:, node.feature] > node.threshold

        # Identify populations for left and right nodes
        left_population = node.sub_population & max_criterion
        right_population = node.sub_population & ~max_criterion

        # Function to check if a node is a leaf
        def is_leaf(population, depth):
            return (depth == self.max_depth - 1) or (np.sum(population) <= self.min_pop) or (np.unique(self.target[population]).size == 1)

        # Process left child
        if is_leaf(left_population, node.depth):
            node.left_child = self.get_leaf_child(node, left_population)
        else:
            node.left_child = self.get_node_child(node, left_population)
            self.fit_node(node.left_child)

        # Process right child
        if is_leaf(right_population, node.depth):
            node.right_child = self.get_leaf_child(node, right_population)
        else:
            node.right_child = self.get_node_child(node, right_population)
            self.fit_node(node.right_child)

    def get_leaf_child(self, node, sub_population):
        """
        Create and return a leaf node.
        The value of the leaf is the most represented class in the sub_population.
        """
        # Determine the most represented class in the sub_population
        value = np.argmax(np.bincount(self.target[sub_population]))
        leaf_child = Leaf(value)
        leaf_child.depth = node.depth + 1
        # NOTE this should be leaf_child.subpopulation_leaf
        leaf_child.subpopulation = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        """
        Create and return a new node with the given sub_population.
        """
        n = Node()
        n.depth = node.depth + 1
        n.sub_population = sub_population
        return n

    def accuracy(self, test_explanatory, test_target):
        """
        Compute the accuracy of the decision tree on the test set.
        """
        predictions = self.predict(test_explanatory)
        return np.sum(np.equal(predictions, test_target)) / test_target.size
