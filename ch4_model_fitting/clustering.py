# 文件功能：
#     1. 从数据集中加载点云数据
#     2. 从点云数据中滤除地面点云
#     3. 从剩余的点云中提取聚类

import numpy as np
import os
import struct
import math
import random
from sklearn import cluster, datasets, mixture
from sklearn.preprocessing import normalize
from itertools import cycle, islice
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d

def vis_point_cloud(points):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([point_cloud])

# 功能：从kitti的.bin格式点云文件中读取点云
# 输入：
#     path: 文件路径
# 输出：
#     点云数组
def read_velodyne_bin(path, vis = False):
    '''
    :param path:
    :return: homography matrix of the point cloud, N*3
    '''
    pc_list = []
    with open(path, 'rb') as f:
        content = f.read()
        pc_iter = struct.iter_unpack('ffff', content)
        for _, point in enumerate(pc_iter):
            # import pdb; pdb.set_trace()
            pc_list.append([point[0], point[1], point[2]])
    points = np.asarray(pc_list, dtype=np.float32)
    if vis:
        vis_point_cloud(points)
    return points

def preprocessing(data):
    z_threshold = (max(data[:,2]) - min(data[:,2])) * 0.3 + min(data[:,2])
    ground_candidate = data[data[:,2] < z_threshold] 
    return ground_candidate

# def fit_plane(p1,p2,p3):
#     normal_vector = np.cross(p3-p1, p2-p1)
#     d = -np.sum(p1*normal_vector)
#     a,b,c = normal_vector
#     import pdb; pdb.set_trace()
#     return a,b,c,d
# 功能：从点云文件中滤除地面点
# 输入：
#     data: 一帧完整点云
# 输出：
#     segmengted_cloud: 删除地面点之后的点云
def ground_segmentation(data, mode= 'RANSAC', threshold  =0.3):
    # 作业1
    # 屏蔽开始
    if mode == 'RANSAC':
        data_list = list(data)
        max_counter = 0
        p = 0.999
        s = 3.0
        max_iter = int(np.log(1 - p) / np.log(1 - pow(0.9,s)))
        for _ in range(max_iter):
            counter = 0
            p1, p2, p3 = random.sample(data_list, k=3)
            while np.linalg.det(np.array([p1,p2,p3])) == 0:
                p1, p2, p3 = random.sample(data_list, k=3)
            # fit the plane w.r.t p1, p2, p3 with normal vector method
            normal_vector= np.cross(p3-p1, p2-p1)
            normal_vector /= np.linalg.norm(normal_vector)
            d = -np.sum(p1*normal_vector)

            distance_list = []
            for d in data:
                distance = np.dot(d - p1, normal_vector)
                if distance < threshold:
                    counter += 1
            if max_counter < counter:
                a_, b_, c_, d_ = normal_vector, d
        return a_, b_, c_, d_

    # 屏蔽结束

    print('origin data points num:', data.shape[0])
    print('segmented data points num:', segmengted_cloud.shape[0])
    return segmengted_cloud

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
    filename = 'datas/000000.bin'
    origin_points = read_velodyne_bin(filename, vis=False)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(origin_points)
    pcd_filtered, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.7)
    pcd_filtered.paint_uniform_color([0,1,0])
    # import pdb;pdb.set_trace()
    possible_ground_points = preprocessing(np.asarray(pcd_filtered.points))
    # pcd_ground = o3d.geometry.PointCloud()
    # pcd_ground.points = o3d.utility.Vector3dVector(possible_ground_points)
    # pcd_ground.paint_uniform_color([0,0,1])
    # o3d.visualization.draw_geometries([pcd_filtered, pcd_ground])
    


    segmented_points = ground_segmentation(data=possible_ground_points)
    cluster_index = clustering(segmented_points)
    plot_clusters(segmented_points, cluster_index)

if __name__ == '__main__':
    main()
