# 文件功能：
#     1. 从数据集中加载点云数据
#     2. 从点云数据中滤除地面点云
#     3. 从剩余的点云中提取聚类

import numpy as np
import open3d as o3d
import math
import os
import random
import struct
from sklearn import cluster, datasets, mixture
from itertools import cycle, islice
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
        for idx, point in enumerate(pc_iter):
            pc_list.append([point[0], point[1], point[2]])
    return np.asarray(pc_list, dtype=np.float32)






# 功能：从点云文件中滤除地面点
# 输入：
#     data: 一帧完整点云
# 输出：
#     segmengted_cloud: 删除地面点之后的点云
def ground_segmentation(data):

    # 作业1
    # 屏蔽开始
    '''z = data[:,-1]
    zmin = min(z)
    zmax = max(z)
    height = zmax - zmin
    ground = []
    for i in range(len(data)):
        threshold = zmin + height * 0.7
        if data[i,-1] < threshold:
            ground.append(data[i])'''

    data_filtered = data #np.asarray(ground)

    ##RANSAC
    max_iter = 30
    consistence = 0
    threshold_ransac = 0.2
    data_list = list(data_filtered)
    plane_param = []
    for i in range(max_iter):
        num = 0
        #random choice 3 points
        s1,s2,s3 = random.sample(data_list, 3)
        if np.linalg.det(np.array([s1,s2,s3])) is 0:
            pass

        #form a plane
        #x * normal_vector + c = 0
        normal_vector = np.cross(s2-s1, s3-s1)
        normal_vector /= np.linalg.norm(normal_vector)
        c = -np.sum(s1*normal_vector)

        #calculate distance
        for j in range(len(data_filtered)):
            d = math.fabs(np.sum(data_list[j]*normal_vector) + c)
            if d < threshold_ransac:
                num += 1
        #count points within a threshold
        if num >= consistence:
            consistence = num
            plane_param = [normal_vector, c]
    print(plane_param)
    # 屏蔽结束
    #delete ground points
    to_be_deleted = []
    threshold_delete = 0.35
    for i in range(len(data_filtered)):
        #calulate distance to the ground
        d = math.fabs(np.sum(data_list[i]*plane_param[0]) + plane_param[1])
        if d < threshold_delete:
            to_be_deleted.append(i)
    to_be_deleted = np.asarray(to_be_deleted)
    segmengted_cloud = np.delete(data, to_be_deleted, axis = 0)

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
    ##DBSCAN
    cur_cloud = o3d.geometry.PointCloud()
    cur_cloud.points = o3d.utility.Vector3dVector(data)
    radius = 10
    min_samples = 200
    visit = np.zeros(len(data))
    clusters = np.zeros(len(data))
    cnum = 1
    pcd_tree = o3d.geometry.KDTreeFlann(cur_cloud)
    for i in range(len(data)):
        if visit[i] is 0
        #find its neighbors' indices within r 
        query = data[i,:]
        [x, idx, _] = pcd_tree.search_radius_vector_3d(query, radius)
        #
        if x > min_samples:
            visit[i] = 1
            clusters[i] = cnum




    # 屏蔽结束

    #return clusters_index'''

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
    root_dir = '/home/gfeng/gfeng_ws/point_cloud_processing/ch4_model_fitting/data' # 数据集路径
    cat = os.listdir(root_dir)
    #cat = cat[1:]
    iteration_num = len(cat)

    for i in range(iteration_num):
        filename = os.path.join(root_dir, cat[i])
        print('clustering pointcloud file:', filename)

        origin_points = read_velodyne_bin(filename)
        segmented_points = ground_segmentation(data = origin_points)
        #cluster_index = clustering(segmented_points)

        #plot_clusters(segmented_points, cluster_index)
        pcd_ground = o3d.geometry.PointCloud()
        pcd_ground.points = o3d.utility.Vector3dVector(segmented_points)       
        o3d.visualization.draw_geometries([pcd_ground])

if __name__ == '__main__':
    main()
