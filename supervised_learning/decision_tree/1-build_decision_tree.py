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
        else:
            left_depth = self.left_child.max_depth_below()\
                         if self.left_child else self.depth
            right_depth = self.right_child.max_depth_below()\
                if self.right_child else self.depth
            return max(left_depth, right_depth)

    def count_nodes_below(self, only_leaves=False):
        """
        We're counting the nodes present
        We have the option to just count the leaves
        """
        ToN = 0 if only_leaves else 1
        if self.left_child:
            ToN += self.left_child.count_nodes_below(only_leaves)
        if self.right_child:
            ToN += self.right_child.count_nodes_below(only_leaves)
        return ToN


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
