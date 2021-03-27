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
    # pcd_filtered.paint_uniform_color([1,0,0])
    # import pdb;pdb.set_trace()
    possible_ground_points = preprocessing(np.asarray(pcd_filtered.points))
    
    pcd_segmented_points = o3d.geometry.PointCloud()
    pcd_segmented_points.points = o3d.utility.Vector3dVector(ground_cloud)
    pcd_segmented_points.paint_uniform_color([0./255.,224./255.,230./255.])
    
    o3d.visualization.draw_geometries([pcd_filtered, pcd_segmented_points])

def preprocessing(data):
    z_threshold = (max(data[:,2]) - min(data[:,2])) * 0.5 + min(data[:,2])
    ground_candidate = data[data[:,2] < z_threshold] 
    return ground_candidate

def ransac(data_list, threshold = 0.1):
    max_counter = 0
    p = 0.9999
    s = 3.0
    max_iter = int(np.log(1 - p) / np.log(1 - pow(0.50, s)))
    print("Max Iteration is ", max_iter)
    
    for _ in range(100):
        
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
            print("max_counter : ", max_counter)
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
    # print("Segmented points num: ", segmented_cloud.shape[0])
    # print("Ground cloud num: ", ground_cloud.shape[0])
    
    return segmented_cloud, ground_cloud

def least_square(ground_cloud, plane_param, max_iter, learning_rate, eps_param):
    A = np.c_[ground_cloud, np.ones((ground_cloud.shape[0],1))]
    learning_rate_backup = copy.deepcopy(learning_rate)
    ATA = A.T @ A 
    # U, S, VT = np.linalg.svd(A, full_matrices = True)
    param = np.array([plane_param]).reshape((4,1))
    for current_iter in range(max_iter):
        learning_rate = copy.deepcopy(learning_rate_backup)
        flag = True
        # import pdb; pdb.set_trace()
        while flag:
            param_candidate = param - (ATA @ param / 1 + np.linalg.norm(A @ param))* learning_rate
            current_loss = np.log(1 + np.linalg.norm(A @ param))
            predicted_loss = np.log(1 + np.linalg.norm(A @ param_candidate))
            if current_loss <= predicted_loss:
                print("Refuse to update, loss increase {}, reduce learning rate...".format(predicted_loss - current_loss))
                learning_rate *= 0.5
            else:
                print("Accept to update, loss reduce {}, learning_rate: {} ".format(predicted_loss - current_loss, learning_rate))
                flag = False
        last_param = copy.deepcopy(param)
        
        param = param_candidate
        param_dist = np.linalg.norm(param - last_param)
        # print("@@@ param_distance: ", param_dist)
        # print("@@@ gradient: ", np.linalg.norm(A.T @ A @ param))
        if param_dist < eps_param and np.linalg.norm(A.T @ A @ param) < 1.0:
            break
    print("Algorithm stops at iteration: ", current_iter)
    return param
# 功能：从点云文件中滤除地面点
# 输入：
#     data: 一帧完整点云
# 输出：
#     segmengted_cloud: 删除地面点之后的点云
def ground_segmentation(data, threshold = 0.15):
    # 作业1
    # 屏蔽开始
    
    data_list = list(data)
    plane_param = ransac(data_list, threshold)
    print("Plane parameter after ransac: ", plane_param)
    
    segmented_cloud, ground_cloud = split_points(data_list, plane_param, 0.3)
    print('segmented data points after ransac num:', segmented_cloud.shape[0])
    print('ground points after ransac num:', data.shape[0] - segmented_cloud.shape[0])
    # vis_ground(data, ground_cloud)
    
    if mode == 'ransac_lsq':
        plane_param = least_square(ground_cloud, plane_param, max_iter = 1000, learning_rate = 1e-6, eps_param = 0.0001)
        segmented_cloud, ground_cloud = split_points(data_list, plane_param, 0.38)
    
    # vis_ground(data, ground_cloud)
    
    print("Plane parameter after least square: ", plane_param[0], plane_param[1], plane_param[2], plane_param[3],)
    # import pdb; pdb.set_trace()
    print('origin data points num:', data.shape[0])
    print('segmented data points after lsq num:', segmented_cloud.shape[0])
    print('ground points after lsq num:', data.shape[0] - segmented_cloud.shape[0])
    return segmented_cloud, ground_cloud