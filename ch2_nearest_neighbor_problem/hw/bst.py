import numpy as np
import matplotlib.pyplot as plt

from result_set import KNNResultSet, RadiusNNResultSet
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


def onenn_search_(node, query_data, min_dist, min_dist_node):
    # print("Min distance is {}, Min distance node is {}".format(min_dist, min_dist_node))

    if node is not None:
        if query_data > node.value:
            if min_dist > query_data - node.value:
                min_dist = query_data - node.value
                min_dist_node = node
            min_dist_node = onenn_search_(node.right, query_data, min_dist, min_dist_node)
        elif query_data < node.value:
            if min_dist > node.value - query_data:
                min_dist = node.value - query_data
                min_dist_node = node
            min_dist_node = onenn_search_(node.left, query_data, min_dist, min_dist_node)
        else: 
            return min_dist_node
    
    return min_dist_node

# do one nn search 
def onenn_search(root, query_data):
    if root is None:
        print("There is no binary search tree.")
        return root

    min_dist = float('inf')
    min_dist_node = root
    result = onenn_search_(root, query_data, min_dist, min_dist_node)
    
    return result


def knn_search_(node, query_data, result_set):

    if node is not None:
        if query_data < node.value:
            result_set = knn_search_(node.left, query_data, result_set)
            if  abs(node.value - query_data) <= result_set.worst_dist:
                result_set.add_point(abs(node.value - query_data), node.index)
                result_set = knn_search_(node.right, query_data, result_set)
        
        elif query_data > node.value:
            result_set = knn_search_(node.right, query_data, result_set)
            if abs(node.value - query_data) <= result_set.worst_dist:
                result_set.add_point(abs(node.value - query_data), node.index)
                result_set = knn_search_(node.left, query_data, result_set)
        
        else:
            result_set.add_point(0,node.index)
            result_set = knn_search_(node.left, query_data, result_set)
            result_set = knn_search_(node.right, query_data, result_set)
    
    return result_set


def knn_search(root, capacity, query_data):
    if root is None:
        print("There is no binary search tree.")
        return root
    
    result_set = KNNResultSet(capacity = capacity)

    result_set = knn_search_(root, query_data, result_set)
    
    return result_set


def radiusnn_search_(node, query_data, result_set):

    if node is not None:
        if query_data < node.value:
            result_set = radiusnn_search_(node.left, query_data, result_set)
            if abs(node.value - query_data) <= result_set.radius:
                result_set.add_point(abs(node.value - query_data), node.index)
                result_set = radiusnn_search_(node.right, query_data, result_set)
        elif query_data > node.value:
            result_set = radiusnn_search_(node.right, query_data, result_set)
            if abs(node.value - query_data) <= result_set.radius:
                result_set.add_point(abs(node.value - query_data), node.index)
                result_set = radiusnn_search_(node.left, query_data, result_set)
        else:
            result_set.add_point(abs(node.value - query_data), node.index)
            result_set = radiusnn_search_(node.left, query_data, result_set)
            result_set = radiusnn_search_(node.right, query_data, result_set)
    return result_set


def radiusnn_search(root, min_radius_dist, query_data):
    if root is None:
        print("There is no binary search tree.")
        return root
    
    result_set = RadiusNNResultSet(min_radius_dist)

    result_set = radiusnn_search_(root, query_data, result_set)
    print(result_set)
    return result_set


def knn_search_lecture(root: BSTNode, value,  result_set: KNNResultSet):
    if root is None:
        return False

    result_set.add_point(abs(root.value - value), root.index)
    if result_set.worstDist() == 0:
        return True

    if root.value >= value:
        if knn_search_lecture(root.left, value, result_set):
            return True
        elif abs(root.value - value) < result_set.worstDist():
            return knn_search_lecture(root.right, value, result_set)
        return False
    
    else:
        if knn_search_lecture(root.right, value, result_set):
            return True
        elif abs(root.value - value) < result_set.worstDist():
            return knn_search_lecture(root.left, value, result_set)
        return False


def radius_search_lecture(root: BSTNode, value,  result_set: KNNResultSet):
    if root is None:
        return False

    result_set.add_point(abs(root.value - value), root.index)

    if root.value >= value:
        if radius_search_lecture(root.left, value, result_set):
            return True
        elif abs(root.value - value) < result_set.worstDist():
            return radius_search_lecture(root.right, value, result_set)
        return False
    
    else:
        if radius_search_lecture(root.right, value, result_set):
            return True
        elif abs(root.value - value) < result_set.worstDist():
            return radius_search_lecture(root.left, value, result_set)
        return False