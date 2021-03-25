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

def vis_ground(data, ground_cloud):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data)
    pcd_filtered, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.7)
    pcd_filtered.paint_uniform_color([1,0,0])
    # import pdb;pdb.set_trace()
    possible_ground_points = preprocessing(np.asarray(pcd_filtered.points))
    
    pcd_segmented_points = o3d.geometry.PointCloud()
    pcd_segmented_points.points = o3d.utility.Vector3dVector(ground_cloud)
    pcd_segmented_points.paint_uniform_color([0,0,0.5])
    
    o3d.visualization.draw_geometries([pcd_filtered, pcd_segmented_points])

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
    z_threshold = (max(data[:,2]) - min(data[:,2])) * 0.4 + min(data[:,2])
    ground_candidate = data[data[:,2] < z_threshold] 
    return ground_candidate

def ransac(data_list, threshold):
    max_counter = 0
    p = 0.999
    s = 3.0
    max_iter = int(np.log(1 - p) / np.log(1 - pow(0.60, s)))
    print("Max Iteration is ", max_iter)
    
    for _ in range(max_iter):
        
        counter = 0
        p1, p2, p3 = random.sample(data_list, k=3)
        while np.linalg.det(np.array([p1,p2,p3])) == 0:
            p1, p2, p3 = random.sample(data_list, k=3)
        # fit the plane w.r.t p1, p2, p3 with normal vector method
        normal_vector= np.cross(p3 - p1, p2 - p1)
        normal_vector /= np.linalg.norm(normal_vector)
        d_ = -np.sum(p1 * normal_vector)
        distance_list = []
        for d in data_list:
            distance = math.fabs(np.dot(d , normal_vector) + d_)
            if distance <= threshold:
                counter += 1
                
        if max_counter < counter:
            
            max_counter = counter
            print("max_counter: ", max_counter)
            a_param, b_param, c_param, d_param = normal_vector[0], normal_vector[1], normal_vector[2], d_
            print("Current plane is : ", a_param, b_param, c_param, d_param)
    return a_param, b_param, c_param, d_param
    
def split_points(data_list, plane_param, threshold):
    a_, b_, c_, d_ = plane_param
    normal_vector = np.array([a_, b_, c_], dtype=np.float64)
    segmented_cloud = []
    ground_cloud =  [] 
    for d in data_list:
        # print("Distance between {} and plane is ".format(data[i]), np.dot(data_list[i], normal_vector) + d_)
        distance = math.fabs(np.dot(d, normal_vector) + d_)
        if distance > threshold:
            # import pdb; pdb.set_trace()
            segmented_cloud.append(d)
        else:
            ground_cloud.append(d)

    segmented_cloud = np.asarray(segmented_cloud)   
    ground_cloud = np.asarray(ground_cloud)     
    print("Segmented points num: ", segmented_cloud.shape[0])
    print("Ground cloud num: ", ground_cloud.shape[0])
    
    return segmented_cloud, ground_cloud

def least_square(ground_cloud, plane_param, max_iter, learning_rate, eps_param):
    A = np.c_[ground_cloud, np.ones((ground_cloud.shape[0],1))]
    ATA = A.T @ A 
    # U, S, VT = np.linalg.svd(A, full_matrices = True)
    param = np.array([plane_param]).reshape((4,1))
    for current_iter in range(max_iter):
        last_param = copy.deepcopy(param)
        param -= (ATA @ param / 1 + np.linalg.norm(A @ param))* learning_rate
        param_dist = np.linalg.norm(param - last_param)
        # print("@@@ param_distance: ", param_dist)
        # print("@@@ gradient: ", np.linalg.norm(A.T @ A @ param))
        if param_dist < eps_param and np.linalg.norm(A.T @ A @ param) < 0.1:
            break
    print("Algorithm stops at iteration: ", current_iter)
    return param
# 功能：从点云文件中滤除地面点
# 输入：
#     data: 一帧完整点云
# 输出：
#     segmengted_cloud: 删除地面点之后的点云
def ground_segmentation(data, threshold = 0.1, mode = 'ransac_lsq'):
    # 作业1
    # 屏蔽开始
    
    data_list = list(data)
    plane_param = ransac(data_list, threshold)
    print("Plane parameter after ransac: ", plane_param)
    
    segmented_cloud, ground_cloud = split_points(data_list, plane_param, 0.3)
    print('segmented data points after ransac num:', segmented_cloud.shape[0])
    print('ground points after ransac num:', data.shape[0] - segmented_cloud.shape[0])
    vis_ground(data, ground_cloud)
    
    if mode == 'ransac_lsq':
        plane_param = least_square(ground_cloud, plane_param, max_iter = 2000, learning_rate = 1e-7, eps_param = 0.0001)
        segmented_cloud, ground_cloud = split_points(data_list, plane_param, 0.25)
    
    vis_ground(data, ground_cloud)
    
    print("Plane parameter after least square: ", plane_param)
    # import pdb; pdb.set_trace()
    print('origin data points num:', data.shape[0])
    print('segmented data points after lsq num:', segmented_cloud.shape[0])
    print('ground points after lsq num:', data.shape[0] - segmented_cloud.shape[0])
    return segmented_cloud, ground_cloud

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
    pcd_filtered.paint_uniform_color([1,0,0])
    # import pdb;pdb.set_trace()
    possible_ground_points = preprocessing(np.asarray(pcd_filtered.points))
    
    segmented_points, ground_cloud = ground_segmentation(data=possible_ground_points)



    pcd_segmented_points = o3d.geometry.PointCloud()
    pcd_segmented_points.points = o3d.utility.Vector3dVector(ground_cloud)
    pcd_segmented_points.paint_uniform_color([0,0,0.5])
    
    o3d.visualization.draw_geometries([pcd_filtered, pcd_segmented_points])

    

    # cluster_index = clustering(segmented_points)
    # plot_clusters(segmented_points, cluster_index)

if __name__ == '__main__':
    main()
