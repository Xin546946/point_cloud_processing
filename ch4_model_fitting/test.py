# 文件功能：
#     1. 从数据集中加载点云数据
#     2. 从点云数据中滤除地面点云
#     3. 从剩余的点云中提取聚类

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
from ground_filter import preprocessing, ground_segmentation, vis_ground

# 功能：从kitti的.bin格式点云文件中读取点云
# 输入：
#     path: 文件路径
# 输出：
#     点云数组
def read_velodyne_bin(path):
    '''
    :param path:
    :return: homography matrix of the point cloud, N*3
    '''
    pc_list = []
    with open(path, 'rb') as f:
        content = f.read()
        pc_iter = struct.iter_unpack('ffff', content)
        for _, point in enumerate(pc_iter):
            pc_list.append([point[0], point[1], point[2]])
    points = np.asarray(pc_list, dtype=np.float32)

    return points





# 功能：显示聚类点云，每个聚类一种颜色
# 输入：
#      data：点云数据（滤除地面后的点云）
#      cluster_index：一维数组，存储的是点云中每个点所属的聚类编号（与上同）
def plot_clusters(data, cluster_index):
    ax = plt.figure().add_subplot(111, projection = '3d')
    colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                             '#f781bf', '#a65628', '#984ea3',
                                             '#999999', '#e41a1c', '#dede00']),
                                      int(max(cluster_index) + 1))))
    colors = np.append(colors, ["#000000"])
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], s=2, color=colors[cluster_index])
    plt.show()

def main():
    filename = 'datas/000001.bin'
    origin_points = read_velodyne_bin(filename)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(origin_points)
    pcd_filtered, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.7)
    # pcd_filtered.paint_uniform_color([1,0,0])
    # import pdb;pdb.set_trace()
    possible_ground_points = preprocessing(np.asarray(pcd_filtered.points))
    segmented_points, ground_cloud = ground_segmentation(data=possible_ground_points)
    
    vis_ground(origin_points, ground_cloud)

    # cluster_index = clustering(segmented_points)
    # plot_clusters(segmented_points, cluster_index)

if __name__ == '__main__':
    main()
