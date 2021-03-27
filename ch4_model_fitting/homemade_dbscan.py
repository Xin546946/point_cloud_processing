
import numpy as np
import open3d as o3d
import math
import os
import struct
import matplotlib.pyplot as plt
import random


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

def custom_draw_geometry_with_key_callback(pcd):
    
    def change_background_to_black(vis):
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0, 0, 0])
        return False
    def change_background_to_white(vis):
        opt = vis.get_render_option()
        opt.background_color = np.asarray([1, 1, 1])
        return False
    
    key_to_callback = {}
    key_to_callback[ord("B")] = change_background_to_black
    key_to_callback[ord("W")] = change_background_to_white
    o3d.visualization.draw_geometries_with_key_callbacks([pcd], key_to_callback)

def ground_segmentation(data):
    
    ##RANSAC
    max_iter = 30
    consistence = 0
    threshold_ransac = 0.2
    data_list = list(data)
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
        for j in range(len(data)):
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
    threshold_delete = 0.4
    for i in range(len(data)):
        #calulate distance to the ground
        d = math.fabs(np.sum(data_list[i]*plane_param[0]) + plane_param[1])
        if d < threshold_delete:
            to_be_deleted.append(i)
    to_be_deleted = np.asarray(to_be_deleted)
    segmengted_cloud = np.delete(data, to_be_deleted, axis = 0)

    print('origin data points num:', data.shape[0])
    print('segmented data points num:', segmengted_cloud.shape[0])
    return segmengted_cloud

class dbscan:
    
    def __init__(self, data, eps, min_points):
        self.min_points = min_points
        self.data = data
        self.eps = eps
        self.visit = np.zeros(len(data))
        self.label = np.zeros(len(data))
        self.cur_cluster = 1
        self.cloud = o3d.geometry.PointCloud()
        self.cloud.points = o3d.utility.Vector3dVector(data)
        self.tree = o3d.geometry.KDTreeFlann(self.cloud)
        self.origin_cloud = o3d.geometry.PointCloud()
        self.origin_cloud.points = o3d.utility.Vector3dVector(data)

    def traverse_neighbors(self, i):
        if self.visit[i] == 0:
            [x, cur_neighbors, _] = self.tree.search_radius_vector_3d(self.data[i,:], self.eps)
            self.visit[i] = 1
            self.label[i] = self.cur_cluster
            #if it is a corepoint
            if x > self.min_points:
                for nid in np.asarray(cur_neighbors):
                    self.traverse_neighbors(nid)

    def clustering(self):
        for i in range(len(self.data)):
            if self.visit[i] == 0:
                [x, neighbors, _] = self.tree.search_radius_vector_3d(self.data[i,:], self.eps)
                self.visit[i] = 1
                if x > self.min_points:
                    self.label[i] = self.cur_cluster
                    for id in np.asarray(neighbors):
                        self.traverse_neighbors(id)
            
                    to_be_deleted = []
                    for j in range(len(self.data)):
                        if self.label[j] == self.cur_cluster:
                            to_be_deleted.append(j)
                    to_be_deleted = np.asarray(to_be_deleted)
                    new_data = np.delete(self.data, to_be_deleted, axis = 0)
                    self.cloud.points = o3d.utility.Vector3dVector(new_data)
                    self.tree = o3d.geometry.KDTreeFlann(self.cloud)    

                self.cur_cluster += 1

        max_label = self.label.max()
        print(f"point cloud has {max_label + 1} clusters")
        colors = plt.get_cmap("tab20")(self.label / (max_label if max_label > 0 else 1))
        colors[self.label < 0] = 0
        self.origin_cloud.colors = o3d.utility.Vector3dVector(colors[:, :3])       

def main():

    root_dir = '/home/gfeng/gfeng_ws/point_cloud_processing/ch4_model_fitting/data' # 数据集路径
    cat = os.listdir(root_dir)
    iteration_num = len(cat)
    for i in range(iteration_num):
        filename = os.path.join(root_dir, cat[i])
    
    data = read_velodyne_bin(filename)
    data = data[0:100,:]

    scan = dbscan(data = data, eps = 10, min_points = 0)
    scan.clustering()
    custom_draw_geometry_with_key_callback(scan.origin_cloud)

if __name__ == '__main__':
    main()