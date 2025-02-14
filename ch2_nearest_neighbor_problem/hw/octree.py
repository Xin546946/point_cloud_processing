# octree的具体实现，包括构建和查找

import random
import math
import numpy as np
import time

from result_set import KNNResultSet, RadiusNNResultSet

# 节点，构成OCtree的基本元素
class Octant:
    def __init__(self, children, center, extent, point_indices, is_leaf):
        self.children = children
        self.center = center
        self.extent = extent
        self.point_indices = point_indices
        self.is_leaf = is_leaf

    def __str__(self):
        output = ''
        output += 'center: [%.2f, %.2f, %.2f], ' % (self.center[0], self.center[1], self.center[2])
        output += 'extent: %.2f, ' % self.extent
        output += 'is_leaf: %d, ' % self.is_leaf
        output += 'children: ' + str([x is not None for x in self.children]) + ", "
        output += 'point_indices: ' + str(self.point_indices)
        return output

# 功能：翻转octree
# 输入：
#     root: 构建好的octree
#     depth: 当前深度
#     max_depth：最大深度
def traverse_octree(root: Octant, depth, max_depth):
    depth[0] += 1
    if max_depth[0] < depth[0]:
        max_depth[0] = depth[0]

    if root is None:
        pass
    elif root.is_leaf:
        print(root)
    else:
        for child in root.children:
            traverse_octree(child, depth, max_depth)
    depth[0] -= 1

# 功能：通过递归的方式构建octree
# 输入：
#     root：根节点
#     db：原始数据
#     center: 中心
#     extent: 当前分割区间
#     point_indices: 点的key
#     leaf_size: scale
#     min_extent: 最小分割区间
def octree_recursive_build(root, db, center, extent, point_indices, leaf_size, min_extent):
    if len(point_indices) == 0:
        return None

    if root is None:
        root = Octant([None for i in range(8)], center, extent, point_indices, is_leaf=True)

    # determine whether to split this octant
    if len(point_indices) <= leaf_size or extent <= min_extent:
        root.is_leaf = True
    else:
        # 作业4
        # 屏蔽开始
        # 如果没满足停止分割的条件，当前octant里的点不是leaf，需要继续分割
        root.is_leaf = False
        # 构造出来一个list，里面有8个小list，分别存放每个octant中的children的index
        children_point_indices = [[] for i in range(8)]
        # 遍历当前octant中所有的点，将其分给上面对应的8个小list里
        for point_idx in point_indices:
            point_db = db[point_idx]
            # 用位运算来控制点应该在哪个octant里
            morton_code = 0
            if point_db[0] > center[0]:
                morton_code = morton_code | 1
            if point_db[1] > center[1]:
                morton_code =morton_code | 2
            if point_db[2] > center[2]:
                morton_code =morton_code | 4
            # 此时已经计算出当前的点应该在第几个list中，加进去
            children_point_indices[morton_code].append(point_idx)
        # 同上，用位运算来确定点到底在哪个octant
        factor = [-0.5, 0.5]
        for i in range(8):
            child_center_x = center[0] + factor[(i & 1) > 0] * extent
            child_center_y = center[1] + factor[(i & 2) > 0] * extent
            child_center_z = center[2] + factor[(i & 4) > 0] * extent
            child_extent = 0.5 * extent
            child_center = np.asarray([child_center_x, child_center_y, child_center_z])
            root.children[i] = octree_recursive_build(root.children[i], 
                                                        db, 
                                                        child_center, 
                                                        child_extent, 
                                                        children_point_indices[i], 
                                                        leaf_size,
                                                        min_extent)
        # 屏蔽结束
    return root

# 功能：判断当前query区间是否在octant内
# 输入：
#     query: 索引信息
#     radius：索引半径
#     octant：octree
# 输出：
#     判断结果，即True/False
def is_ball_in_octant(query: np.ndarray, radius: float, octant:Octant):
    """
    Determines if the query ball is inside the octant
    :param query:
    :param radius:
    :param octant:
    :return:
    """
    query_offset = query - octant.center
    query_offset_abs = np.fabs(query_offset)
    possible_space = query_offset_abs + radius
    # 如果possible_space里的元素都比对应位置的extent小，则说明query为中心，radius为半径的球在octant里
    return np.all(possible_space < octant.extent)

# 功能：判断当前query区间是否和octant有重叠部分
# 输入：
#     query: 索引信息
#     radius：索引半径
#     octant：octree
# 输出：
#     判断结果，即True/False
def overlaps(query: np.ndarray, radius: float, octant:Octant):
    """
    Determines if the query ball overlaps with the octant
    :param query:
    :param radius:
    :param octant:
    :return:
    """
    query_offset = query - octant.center
    query_offset_abs = np.fabs(query_offset)

    # completely outside, since query is outside the relevant area
    max_dist = radius + octant.extent
    if np.any(query_offset_abs > max_dist):
        return False

    # if pass the above check, consider the case that the ball is contacting the face of the octant
    if np.sum((query_offset_abs < octant.extent).astype(np.int)) >= 2:
        return True

    # conside the case that the ball is contacting the edge or corner of the octant
    # since the case of the ball center (query) inside octant has been considered,
    # we only consider the ball center (query) outside octant
    x_diff = max(query_offset_abs[0] - octant.extent, 0)
    y_diff = max(query_offset_abs[1] - octant.extent, 0)
    z_diff = max(query_offset_abs[2] - octant.extent, 0)

    return x_diff * x_diff + y_diff * y_diff + z_diff * z_diff < radius * radius


# 功能：判断当前query是否包含octant
# 输入：
#     query: 索引信息
#     radius：索引半径
#     octant：octree
# 输出：
#     判断结果，即True/False
def is_octant_in_ball(query: np.ndarray, radius: float, octant:Octant):
    """
    Determine if the query ball contains the octant
    :param query:
    :param radius:
    :param octant:
    :return:
    """
    query_offset = query - octant.center
    query_offset_abs = np.fabs(query_offset)

    query_offset_to_farthest_corner = query_offset_abs + octant.extent
    return np.linalg.norm(query_offset_to_farthest_corner) < radius

# 功能：在octree中查找信息
# 输入：
#    root: octree
#    db：原始数据
#    result_set: 索引结果
#    query：索引信息
def octree_radius_search_fast(root: Octant, db: np.ndarray, result_set: RadiusNNResultSet, query: np.ndarray):
    if root is None:
        return False

    # 作业5
    # 提示：尽量利用上面的is_ball_in_octant、overlaps、is_octant_in_ball等函数
    # 屏蔽开始
    if is_octant_in_ball(query, result_set.worstDist(), root):
        leaf_points = db[root.point_indices, :]
        diff = np.linalg.norm(np.expand_dims(query, 0) - leaf_points, axis = 1)
        for i in range(diff.shape[0]):
            result_set.add_point(diff[i], root.point_indices[i])
        return False
    
    if root.is_leaf and len(root.point_indices) > 0:
        leaf_points = db[root.point_indices, :]
        diff = np.linalg.norm(np.expand_dims(query, 0) - leaf_points, axis = 1)
        for i in range(diff.shape[0]):
            result_set.add_point(diff[i], root.point_indices[i])
        return is_ball_in_octant(query, result_set.worstDist(), root)
    
    for c, child in enumerate(root.children):
        if child is None:
            continue
        if overlaps(query, result_set.worstDist(), child) == False:
            continue
        if octree_radius_search_fast(child, db, result_set, query):
            return True
        # 屏蔽结束

    return is_ball_in_octant(query, result_set.worstDist(), root)


# 功能：在octree中查找radius范围内的近邻
# 输入：
#     root: octree
#     db: 原始数据
#     result_set: 搜索结果
#     query: 搜索信息
def octree_radius_search(root: Octant, db: np.ndarray, result_set: RadiusNNResultSet, query: np.ndarray):
    if root is None:
        return False

    if root.is_leaf and len(root.point_indices) > 0:
        # compare the contents of a leaf
        leaf_points = db[root.point_indices, :]
        # dim of query (3,), if expand_dims--> (3,1)
        diff = np.linalg.norm(np.expand_dims(query, 0) - leaf_points, axis=1)
        for i in range(diff.shape[0]):
            result_set.add_point(diff[i], root.point_indices[i])
        # check whether we can stop search now
        return is_ball_in_octant(query, result_set.worstDist(), root)

    # 作业6
    # 屏蔽开始
    morton_code = 0
    if query[0] > root.center[0]:
        morton_code = morton_code | 1
    if query[1] > root.center[1]:
        morton_code =morton_code | 2
    if query[2] > root.center[2]:
        morton_code =morton_code | 4
    
    if octree_radius_search(root.children[morton_code],db, result_set, query):
        return True

    for c, child in enumerate(root.children):
        if c == morton_code or child is None:
            continue
        if overlaps(query, result_set.worstDist(), child) == False:
            continue
        if octree_radius_search(child, db, result_set, query):
            return True

    # 屏蔽结束

    # final check of if we can stop search
    return is_ball_in_octant(query, result_set.worstDist(), root)

# 功能：在octree中查找最近的k个近邻
# 输入：
#     root: octree
#     db: 原始数据
#     result_set: 搜索结果
#     query: 搜索信息
def octree_knn_search(root: Octant, db: np.ndarray, result_set: KNNResultSet, query: np.ndarray):
    if root is None:
        return False

    if root.is_leaf and len(root.point_indices) > 0:
        # compare the contents of a leaf
        leaf_points = db[root.point_indices, :]
        diff = np.linalg.norm(np.expand_dims(query, 0) - leaf_points, axis=1)
        for i in range(diff.shape[0]):
            result_set.add_point(diff[i], root.point_indices[i])
        # check whether we can stop search now
        return is_ball_in_octant(query, result_set.worstDist(), root)

    # 作业7
    # 屏蔽开始
    # 如果不是leaf，则需要看当前的query在octant中的哪个child里
    morton_code = 0
    if query[0] > root.center[0]:
        morton_code = morton_code | 1
    if query[1] > root.center[1]:
        morton_code = morton_code | 2
    if query[2] > root.center[2]:
        morton_code = morton_code | 4

    if octree_knn_search(root.children[morton_code], db, result_set, query):
        return True
    
    # 看看最坏距离是否包含其他的octant
    for c, child in enumerate(root.children):
        if c == morton_code or child is None:
            continue
        if overlaps(query, result_set.worstDist(), child) == False:
            continue
        if octree_knn_search(child, db, result_set, query):
            return True
    # 屏蔽结束

    # final check of if we can stop search
    return is_ball_in_octant(query, result_set.worstDist(), root)

# 功能：构建octree，即通过调用octree_recursive_build函数实现对外接口
# 输入：
#    dp_np: 原始数据
#    leaf_size：scale
#    min_extent：最小划分区间
def octree_construction(db_np, leaf_size, min_extent):
    N, dim = db_np.shape[0], db_np.shape[1]
    db_np_min = np.amin(db_np, axis=0)
    db_np_max = np.amax(db_np, axis=0)
    db_extent = np.max(db_np_max - db_np_min) * 0.5
    db_center = np.mean(db_np, axis=0)

    root = None
    root = octree_recursive_build(root, db_np, db_center, db_extent, list(range(N)),
                                  leaf_size, min_extent)

    return root

def main():
    # configuration
    db_size = 64000
    dim = 3
    leaf_size = 4
    min_extent = 0.0001
    k = 8

    db_np = np.random.rand(db_size, dim)

    root = octree_construction(db_np, leaf_size, min_extent)

    depth = [0]
    max_depth = [0]
    traverse_octree(root, depth, max_depth)
    print("tree max depth: %d" % max_depth[0])

    query = np.asarray([0, 0, 0])

    begin_t = time.time()
    result_set = KNNResultSet(capacity=k)
    octree_knn_search(root, db_np, result_set, query)
    print("KNN Search takes %.3fms\n" % ((time.time() - begin_t) * 1000))
    print(result_set)

    begin_t = time.time()
    diff = np.linalg.norm(np.expand_dims(query, 0) - db_np, axis=1)
    nn_idx = np.argsort(diff)
    nn_dist = diff[nn_idx]
    print("Brute Force takes %.3fms\n" % ((time.time() - begin_t) * 1000))
    print(nn_idx[0:k])
    print(nn_dist[0:k])

    np.random.seed(0)

    begin_t = time.time()
    print("Radius search normal:")
    for i in range(100):
        query = np.random.rand(3)
        result_set = RadiusNNResultSet(radius=0.05)
        octree_radius_search(root, db_np, result_set, query)
    print("Search takes %.3fms\n" % ((time.time() - begin_t) * 1000))
    # print(result_set)

    begin_t = time.time()
    print("Radius search fast:")
    for i in range(100):
        query = np.random.rand(3)
        result_set = RadiusNNResultSet(radius = 0.05)
        octree_radius_search_fast(root, db_np, result_set, query)
    # print(result_set)
    print("Search takes %.3fms\n" % ((time.time() - begin_t) * 1000))



if __name__ == '__main__':
    main()