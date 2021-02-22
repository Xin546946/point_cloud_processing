import numpy as np
import matplotlib.pyplot as plt

class BSTNode:
    
    def __init__(self, index, value):
        self.left = None
        self.right = None
        self.index = index # 表示在原先序列中的第几个点
        self.value = value # 
        self.num = 1
    
    def __str__(self):
        return "index: %s, value: %s, num: %s" % (str(self.index), str(self.value), str(self.num))


def construct_bst(data, method):
    if len(data) == 0:
        print("There is no data")
        exit()

    root = None
    if method == 'recursive':
        for i, d in enumerate(data):        
            root = add_data_recursively(root, i, d)
    elif method == 'iterative':
        for i, d in enumerate(data):        
            root = add_data_iteratively(root, i, d)
    
    return root

# todo this written style is so ugly, how to make it beautiful
def add_data_iteratively(root, index, value):
    if root is None:
        root = BSTNode(index,value)
    else:
        current_node = root
        while current_node is not None:
            prev_node = current_node
            if value < current_node.value:
                current_node = current_node.left
                if current_node is None:
                    prev_node.left = BSTNode(index, value)
            elif value > current_node.value:
                current_node = current_node.right
                if current_node is None:
                    prev_node.right = BSTNode(index, value)
            else:
                current_node.num = current_node.num + 1
                break

    return root 


def add_data_recursively(root, index, value):
    if root is None:
        root = BSTNode(index,value)
    else:
        if value < root.value:
            root.left = add_data_recursively(root.left, index, value)
        elif value > root.value:
            root.right = add_data_recursively(root.right, index, value)
        else:
            root.num = root.num + 1 
    return root


def inorder(root):
    if root is not None:
        inorder(root.left)
        print(root)
        inorder(root.right)


def preorder(root):
    if root is not None:
        print(root)
        preorder(root.left)
        preorder(root.right)

def postorder(root):
    if root is not None:
        postorder(root.left)
        postorder(root.right)
        print(root)

def insert(root, index, value):
    if root is None:
        root = BSTNode(index, value)
    else:
        if value < root.value:
            root.left = insert(root.left, index, value)
        elif value > root.value:
            root.right = insert(root.right, index, value)
        else:  # don't insert if key already exist in the tree
            pass
    return root