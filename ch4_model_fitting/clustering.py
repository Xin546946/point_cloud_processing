import numpy as np
import os
import struct
import math
import copy
import random
from sklearn import cluster, datasets, mixture
from sklearn.preprocessing import normalize
from itertools import cycle, islice
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d
from test import read_velodyne_bin

def dbscan(data, distance = 0.3, min_samples = 15):
    points = list(copy.deepcopy(data))
    clusters = np.zeros(len(points)).reshape(len(points),1)
    current_cluster = 0
    visited = list(np.zeros(len(points)))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    import pdb; pdb.set_trace()
    for i in range(len(points)):
        if visited[i] == 0:
            build_current_cluster = True
            while build_current_cluster:
            num, indices, _ = pcd_tree.search_radius_vector_3d(points[i], radius = distance)
            if num >= min_samples:
                visited[i] = 1
                
                num, indices, _ = pcd_tree.search_radius_vector_3d(points[i], radius = distance)
                
                





# 功能：从点云中提取聚类
# 输入：
#     data: 点云（滤除地面后的点云）
# 输出：
#     clusters_index： 一维数组，存储的是点云中每个点所属的聚类编号（参考上一章内容容易理解）
def clustering(data):
    # 作业2
    # 屏蔽开始


    # 屏蔽结束

    return clusters_index

def main():
    filename = 'datas/000001.bin'
    origin_points = read_velodyne_bin(filename)
    dbscan(origin_points)

if __name__ == '__main__':
    main()