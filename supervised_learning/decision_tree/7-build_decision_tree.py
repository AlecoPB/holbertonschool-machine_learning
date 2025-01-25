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

    def fit_node(self, node):
        """
        Fit a single node of the decision tree. Split the population of the node
        into left and right subpopulations based on the splitting criterion.
        """
        # Determine the best splitting feature and threshold for the node
        node.feature, node.threshold = self.split_criterion(node)

        # Split the population into left and right based on the threshold
        left_population = node.sub_population[node.sub_population[:, node.feature] > node.threshold]
        right_population = node.sub_population[node.sub_population[:, node.feature] <= node.threshold]

        # Check if the left child is a leaf
        is_left_leaf = (
            left_population.shape[0] < self.min_pop or
            node.depth + 1 >= self.max_depth or
            np.all(left_population[:, -1] == left_population[0, -1])
        )

        if is_left_leaf:
            node.left_child = self.get_leaf_child(node, left_population)
        else:
            node.left_child = self.get_node_child(node, left_population)
            self.fit_node(node.left_child)

        # Check if the right child is a leaf
        is_right_leaf = (
            right_population.shape[0] < self.min_pop or
            node.depth + 1 >= self.max_depth or
            np.all(right_population[:, -1] == right_population[0, -1])
        )

        if is_right_leaf:
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
        unique_classes, counts = np.unique(sub_population[:, -1], return_counts=True)
        value = unique_classes[np.argmax(counts)]  # Class with the highest count

        # Create and return the leaf node
        leaf_child = Leaf(value)
        leaf_child.depth = node.depth + 1
        leaf_child.sub_population = sub_population
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
