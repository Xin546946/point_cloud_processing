import numpy as np
import result_set as rs

class Node:
    def __init__(self, val, index):
        self.left = None
        self.right = None
        self.val = val
        self.index = index
    def __str__(self):
        return "index:%s, value:%s"%(str(self.index),str(self.val))

def add_node_recursively(root, num, index):
    if root is None :
        root = Node(num,index)
    else:
        if num < root.val:
            root.left = add_node_recursively(root.left, num, index)
        elif num > root.val:
            root.right = add_node_recursively(root.right, num, index)
        else:
            pass
    return root

def build_bst_recursively(data):
    if len(data) is 0:
        return None
    root = None
    for i, num in enumerate(data):
        root = add_node_recursively(root, num, i)
    return root

def add_node_iteratively(root, num, i):
    if root is None:
        root = Node(num, i)
    else:
        cur_node = root
        while cur_node is not None:
            pre = cur_node
            if num < cur_node.val:
                cur_node = cur_node.left
                if(cur_node is None):
                    pre.left = Node(num, i)
            elif num > cur_node.val:
                cur_node = cur_node.right
                if(cur_node is None):
                    pre.right = Node(num, i)
            else:
                break
    return root

def build_bst_iteratively(data):
    if len(data) is 0:
        return None

    root = None

    for i, num in enumerate(data):
        root = add_node_iteratively(root, num, i)
    return root

def inorder(root):
    if root is None:
        return
    inorder(root.left)
    print(root)
    inorder(root.right)

def preorder(root):
    if root is None:
        return
    print(root)
    inorder(root.left)
    inorder(root.right)

def postorder(root):
    if root is None:
        return
    inorder(root.left)
    inorder(root.right)
    print(root)

def o_search(cur_node, min_dist, min_node, quary):
    if cur_node is None:
        return min_node
    if quary is cur_node.val:
        return cur_node

    if quary > cur_node.val:
        if min_dist >= quary - cur_node.val:
            min_dist = quary - cur_node.val
            min_node = cur_node
        min_node = o_search(cur_node.right, min_dist, min_node, quary)
    else:
        if min_dist >= cur_node.val - quary:
            min_dist = cur_node.val - quary
            min_node = cur_node
        min_node = o_search(cur_node.left, min_dist, min_node, quary)

    return min_node

def one_nn_search(root, quary):
    min_dist = float('inf')
    min_node = o_search(root, min_dist, None, quary)
    return min_node

def KNNsearch_step(cur, quary, res_set):

    if cur is not None:
        res_set.add_point(abs(quary - cur.val), cur.index)
        if quary >= cur.val:
            res_set = KNNsearch_step(cur.right, quary, res_set)
            if quary - cur.val <= res_set.worstDist():
                res_set = KNNsearch_step(cur.left, quary, res_set)
        elif quary < cur.val:
            res_set = KNNsearch_step(cur.left, quary, res_set)
            if cur.val - quary <= res_set.worstDist():
                res_set = KNNsearch_step(cur.right, quary, res_set)
    return res_set

def KNNsearch(root, quary, cap):
    if root is None:
        return root
    res = rs.KNNResultSet(cap)
    res = KNNsearch_step(root, quary, res)
    return res

def RadiusNNsearch_step(cur, quary, res_set):

    if cur is not None:
        res_set.add_point(abs(quary - cur.val), cur.index)
        if quary >= cur.val:
            res_set = KNNsearch_step(cur.right, quary, res_set)
            if quary - cur.val <= res_set.worstDist():
                res_set = KNNsearch_step(cur.left, quary, res_set)
        elif quary < cur.val:
            res_set = KNNsearch_step(cur.left, quary, res_set)
            if cur.val - quary <= res_set.worstDist():
                res_set = KNNsearch_step(cur.right, quary, res_set)
    return res_set

def RadiusNNsearch(root, quary, r):
    if root is None:
        return root
    res = rs.RadiusNNResultSet(r)
    res = RadiusNNsearch_step(root, quary, res)
    return res
