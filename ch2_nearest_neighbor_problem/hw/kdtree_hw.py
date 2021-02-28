import numpy as np


class KDTreeNode:
    def __init__(self, value, axis):
        self.axis = axis
        self.value = value
        self.left = None
        self.right = None
        self.children = None
    
    def has_leaves():
        if self.children == None:
            return False
        else:
            return True

def construct_kdtree_recursive(data, leaf_size):
    root = KDTreeNode(None, 0)
    kdtree_root = build_tree(root, data[0], data[-1])
    return kdtree_root


            
