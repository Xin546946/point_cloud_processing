# kdtree的具体实现，包括构建和查找
from scipy.spatial import KDTree
import random
import math
import numpy as np
import time

import datetime

from result_set import KNNResultSet, RadiusNNResultSet

# Node类，Node是tree的基本组成元素
class Node:
    def __init__(self, axis, value, left, right, point_indices):
        self.axis = axis
        self.value = value
        self.left = left
        self.right = right
        self.point_indices = point_indices

    def is_leaf(self):
        if self.value is None:
            return True
        else:
            return False

    def __str__(self):
        output = ''
        output += 'axis %d, ' % self.axis
        if self.value is None:
            output += 'split value: leaf, '
        else:
            output += 'split value: %.2f, ' % self.value
        output += 'point_indices: '
        output += str(self.point_indices.tolist())
        return output

# 功能：构建树之前需要对value进行排序，同时对一个的key的顺序也要跟着改变
# 输入：
#     key：键
#     value:值
# 输出：
#     key_sorted：排序后的键
#     value_sorted：排序后的值
def sort_value_by_index(value, index):
    assert value.shape == index.shape
    assert len(value.shape) == 1
    sorted_idx = np.argsort(index)
    value_sorted = value[sorted_idx]
    index_sorted = index[sorted_idx]
    return value_sorted, index_sorted


def axis_round_robin(axis, dim):
    if axis == dim-1:
        return 0
    else:
        return axis + 1

# 功能：通过递归的方式构建树
# 输入：
#     root: 树的根节点
#     db: 点云数据
#     point_indices：排序后的键
#     axis: scalar
#     leaf_size: scalar
# 输出：
#     root: 即构建完成的树
def kdtree_recursive_build(root, db, point_indices, axis, leaf_size):
    if root is None:
        root = Node(axis, None, None, None, point_indices)

    # determine whether to split into left and right
    if len(point_indices) > leaf_size:
        # --- get the split position ---
        # length_sub_points =  int(len(point_indices) * 0.1)
        # sub_points = random.sample(list(point_indices), length_sub_points)
        # point_indices_sorted, _ = sort_value_by_index(np.asarray(sub_points), db[sub_points, axis])  # M
        point_indices_sorted, _ = sort_value_by_index(np.asarray(point_indices), db[point_indices, axis])
        # 作业1
        # 屏蔽开始
        middle_left_idx = math.ceil(point_indices_sorted.shape[0] / 2) - 1
        middle_left_point_idx = point_indices_sorted[middle_left_idx]
        middle_left_point_value = db[middle_left_point_idx, axis]

        middle_right_idx = middle_left_idx + 1
        middle_right_point_idx = point_indices_sorted[middle_right_idx]
        middle_right_point_value = db[middle_right_point_idx, axis]

        root.value = (middle_left_point_value + middle_right_point_value) * 0.5

        root.left = kdtree_recursive_build(root.left,
                                           db,
                                           point_indices_sorted[0:middle_right_idx],
                                           axis_round_robin(axis, dim=db.shape[1]),
                                           leaf_size)

        root.right = kdtree_recursive_build(root.right,
                                            db,
                                            point_indices_sorted[middle_right_idx:],
                                            axis_round_robin(axis, dim=db.shape[1]),
                                            leaf_size)
        
    # return root

    # 屏蔽结束
    return root


# 功能：翻转一个kd树
# 输入：
#     root：kd树
#     depth: 当前深度
#     max_depth：最大深度
def traverse_kdtree(root: Node, depth, max_depth):
    depth[0] += 1
    if max_depth[0] < depth[0]:
        max_depth[0] = depth[0]

    if root.is_leaf():
        print(root)
        pass
    else:
        traverse_kdtree(root.left, depth, max_depth)
        traverse_kdtree(root.right, depth, max_depth)

    depth[0] -= 1

# 功能：构建kd树（利用kdtree_recursive_build功能函数实现的对外接口）
# 输入：
#     db_np：原始数据
#     leaf_size：scale
# 输出：
#     root：构建完成的kd树
def kdtree_construction(db_np, leaf_size):
    N, dim = db_np.shape[0], db_np.shape[1]

    # build kd_tree recursively
    root = None
    root = kdtree_recursive_build(root,
                                  db_np,
                                  np.arange(N),
                                  axis=0,
                                  leaf_size=leaf_size)
    return root


# 功能：通过kd树实现knn搜索，即找出最近的k个近邻
# 输入：
#     root: kd树
#     db: 原始数据
#     result_set：搜索结果
#     query：索引信息
# 输出：
#     搜索失败则返回False
def kdtree_knn_search(root: Node, db: np.ndarray, result_set: KNNResultSet, query: np.ndarray):
    if root is None:
        return False

    if root.is_leaf():
        # compare the contents of a leaf
        leaf_points = db[root.point_indices, :]
        diff = np.linalg.norm(np.expand_dims(query, 0) - leaf_points, axis=1)
        for i in range(diff.shape[0]):
            result_set.add_point(diff[i], root.point_indices[i])
        return False

    # 作业2
    # 提示：仍通过递归的方式实现搜索
    # 屏蔽开始
    if query[root.axis] <= root.value:
        kdtree_knn_search(root.left, db, result_set, query)
        if math.fabs(query[root.axis] - root.value) < result_set.worstDist():
            kdtree_knn_search(root.right, db, result_set, query)
    else:
        kdtree_knn_search(root.right, db, result_set, query)
        if math.fabs(query[root.axis] - root.value) < result_set.worstDist():
            kdtree_knn_search(root.left, db, result_set, query)

    # 屏蔽结束

    return False

# 功能：通过kd树实现radius搜索，即找出距离radius以内的近邻
# 输入：
#     root: kd树
#     db: 原始数据
#     result_set:搜索结果
#     query：索引信息
# 输出：
#     搜索失败则返回False
def kdtree_radius_search(root: Node, db: np.ndarray, result_set: RadiusNNResultSet, query: np.ndarray):
    if root is None:
        return False

    if root.is_leaf():
        # compare the contents of a leaf
        leaf_points = db[root.point_indices, :]
        diff = np.linalg.norm(np.expand_dims(query, 0) - leaf_points, axis=1)
        for i in range(diff.shape[0]):
            result_set.add_point(diff[i], root.point_indices[i])
        return False
    
    # 作业3
    # 提示：通过递归的方式实现搜索
    # 屏蔽开始
    if query[root.axis] <= root.value:
        kdtree_radius_search(root.left, db, result_set, query)
        if math.fabs(query[root.axis] - root.value) < result_set.worstDist():
            kdtree_radius_search(root.right, db, result_set, query)
    else:
        kdtree_radius_search(root.right, db, result_set, query)
        if math.fabs(query[root.axis] - root.value) < result_set.worstDist():
            kdtree_radius_search(root.left, db, result_set, query)
    # 屏蔽结束

    return False


def main():
    db_size = 100
    dim = 3
    k = 8
    # generate point clouds
    db_np = np.random.rand(db_size, dim)

    # generate query point
    query = np.array([0,0,0])
    # use scipylearn to test knn k = 8

    tree = KDTree(db_np, leafsize = 10)
   
    # begin_t = time.time()
    # knn_result = tree.query(query, k = k)
    # print("KNN Search using scipy takes %.3fms\n" % ((time.time() - begin_t)*1000))

    # print(db_np[knn_result[1]])

    kdtree_root = kdtree_construction(db_np, leaf_size=10)
    traverse_kdtree(kdtree_root,[0], [0])
    # result_set = KNNResultSet(capacity=k)
    # begin_t = time.time()
    # kdtree_knn_search(kdtree_root, db_np, result_set, query)
    # print("KNN Search using my KDTree takes %.3fms\n " % ((time.time() - begin_t) * 1000))

if __name__ == '__main__':
    main()