# 对数据集中的点云，批量执行构建树和查找，包括kdtree和octree，并评测其运行时间

import random
import math
import numpy as np
import time
import os
import struct
import open3d as o3d
from scipy.spatial import KDTree

import open3d as o3d
from pyntcloud import PyntCloud
import octree as octree
import kdtree as kdtree
from result_set import KNNResultSet, RadiusNNResultSet

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

def vis_point_cloud(points):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([point_cloud])
    


def main():
    # configuration
    leaf_size = 32
    min_extent = 0.0001
    k = 8
    radius = 1

    root_dir = '/home/kit/point_cloud_processing/ch2_nearest_neighbor_problem/data' # 数据集路径
    cat = os.listdir(root_dir)
    iteration_num = len(cat)
    filename = os.path.join(root_dir, cat[0])
    db_np = read_velodyne_bin(filename)
    print("Let us see all points")
    for i in range(db_np.shape[0]):
        print(db_np[i])
    
    
    print("My octree --------------")
    construction_time_sum = 0
    knn_time_sum = 0
    radius_time_sum = 0
    brute_time_sum = 0
    for i in range(iteration_num):
        filename = os.path.join(root_dir, cat[i])
        db_np = read_velodyne_bin(filename)

        begin_t = time.time()
        root = octree.octree_construction(db_np, leaf_size, min_extent)
        construction_time_sum += time.time() - begin_t

        query = db_np[0,:]

        begin_t = time.time()
        result_set = KNNResultSet(capacity=k)
        octree.octree_knn_search(root, db_np, result_set, query)
        knn_time_sum += time.time() - begin_t

        begin_t = time.time()
        result_set = RadiusNNResultSet(radius=radius)
        octree.octree_radius_search_fast(root, db_np, result_set, query)
        radius_time_sum += time.time() - begin_t

        begin_t = time.time()
        diff = np.linalg.norm(np.expand_dims(query, 0) - db_np, axis=1)
        nn_idx = np.argsort(diff)
        nn_dist = diff[nn_idx]
        brute_time_sum += time.time() - begin_t
    print("Octree: build %.3fms, knn %.3fms, radius %.3fms, brute %.3fms" % (construction_time_sum*1000/iteration_num,
                                                                     knn_time_sum*1000/iteration_num,
                                                                     radius_time_sum*1000/iteration_num,
                                                                     brute_time_sum*1000/iteration_num))

    print("My kdtree -------leaf_size{}-------".format(leaf_size))
    construction_time_sum = 0
    knn_time_sum = 0
    radius_time_sum = 0
    brute_time_sum = 0
    sklearn_knn_time_sum = 0
    sklearn_construction_time_sum = 0
    
    for i in range(iteration_num):
        filename = os.path.join(root_dir, cat[i])
        db_np = read_velodyne_bin(filename)
        # vis_point_cloud(db_np)
        begin_t = time.time()
        root = kdtree.kdtree_construction(db_np, leaf_size)
        construction_time_sum += time.time() - begin_t

        query = db_np[0,:]

        begin_t = time.time()
        result_set = KNNResultSet(capacity=k)
        kdtree.kdtree_knn_search(root, db_np, result_set, query)
        knn_time_sum += time.time() - begin_t

        begin_t = time.time()
        result_set = RadiusNNResultSet(radius=radius)
        kdtree.kdtree_radius_search(root, db_np, result_set, query)
        radius_time_sum += time.time() - begin_t

        begin_t = time.time()
        diff = np.linalg.norm(np.expand_dims(query, 0) - db_np, axis=1)
        nn_idx = np.argsort(diff)
        nn_dist = diff[nn_idx]
        brute_time_sum += time.time() - begin_t

    print("Kdtree: build %.3fms, knn %.3fms, radius %.3fms, brute %.3fms" % (construction_time_sum * 1000 / iteration_num,
                                                                     knn_time_sum * 1000 / iteration_num,
                                                                     radius_time_sum * 1000 / iteration_num,
                                                                     brute_time_sum * 1000 / iteration_num))
    print("")
    print("open3d kdtree ----------")
    for i in range(iteration_num):
        filename = os.path.join(root_dir, cat[i])
        point_cloud_file = os.path.join(root_dir,"000000.bin")
        point_cloud_pynt = PyntCloud.from_file(point_cloud_file)
        point_cloud_o3d = point_cloud_pynt.to_instance("open3d", mesh=False)

        begin_t = time.time()
        pcd_tree = o3d.geometry.KDTreeFlann(point_cloud_o3d)
        open3d_build_time = time.time() - begin_t

        points = point_cloud_pynt.points

        begin_t = time.time()
        # for id in range(points.shape[0]):
        #     pcd_tree.search_knn_vector_3d(point_cloud_o3d.points[id], k)
        pcd_tree.search_knn_vector_3d(query, k)
        open3d_knn_time = time.time() - begin_t

        begin_t = time.time()
        # for id in range(points.shape[0]):
            # pcd_tree.search_radius_vector_3d(point_cloud_o3d.points[id], radius)
        pcd_tree.search_radius_vector_3d(query, radius)
        open3d_rnn_time = time.time() - begin_t
        

    print("Kdtree: build %.3fms, knn %.3fms, radius %.3fms" % (open3d_build_time * 1000 ,
                                                                     open3d_knn_time * 1000,
                                                                     open3d_rnn_time * 1000))

    sklearn_rnn_time_sum = 0

    print("")
    print("scipy.spatial kdtree ----------")
    for i in range(iteration_num):
        filename = os.path.join(root_dir, cat[i])
        db_np = read_velodyne_bin(filename)
        # vis_point_cloud(db_np)

        query = db_np[0,:]

        begin_t = time.time()
        pcd_tree = o3d.geometry.KDTreeFlann(point_cloud_o3d)
        open3d_build_time = time.time() - begin_t

        points = point_cloud_pynt.points

        begin_t = time.time()
        tree = KDTree(db_np, leafsize = leaf_size)
        sklearn_construction_time_sum += time.time() - begin_t

        begin_t = time.time()
        knn_result = tree.query(query, k = k)
        sklearn_knn_time_sum += time.time() - begin_t
        
        begin_t = time.time()
        knn_result = tree.query_ball_point(query, r = radius)
        sklearn_rnn_time_sum += time.time() - begin_t

    print("Kdtree: build %.3fms, knn %.3fms, radius %.3fms" % ( sklearn_construction_time_sum * 1000 / iteration_num,
                                                                     sklearn_knn_time_sum * 1000 / iteration_num,
                                                                     sklearn_rnn_time_sum * 1000 / iteration_num))

    # print("")
    # print("open3d octree--------")
    # for i in range(iteration_num):
    #     filename = os.path.join(root_dir, cat[i])
    #     point_cloud_file = os.path.join(root_dir,"000000.bin")
    #     point_cloud_pynt = PyntCloud.from_file(point_cloud_file)
    #     point_cloud_o3d = point_cloud_pynt.to_instance("open3d", mesh=False)

    #     begin_t = time.time()
    #     pcd_octree = o3d.geometry.Octree(max_depth=8)
    #     pcd_octree.convert_from_point_cloud(point_cloud_o3d, size_expand=0.01)
        
    #     open3d_build_time = time.time() - begin_t

    #     points = point_cloud_pynt.points

    #     begin_t = time.time()
    #     # for id in range(points.shape[0]):
    #     #     pcd_tree.search_knn_vector_3d(point_cloud_o3d.points[id], k)
    #     pcd_tree.search_knn_vector_3d(query, k)
    #     open3d_knn_time = time.time() - begin_t

    #     begin_t = time.time()
    #     # for id in range(points.shape[0]):
    #         # pcd_tree.search_radius_vector_3d(point_cloud_o3d.points[id], radius)
    #     pcd_tree.search_radius_vector_3d(query, radius)
    #     open3d_rnn_time = time.time() - begin_t
        

    # print("Kdtree: build %.3fms, knn %.3fms, radius %.3fms" % (open3d_build_time * 1000 ,
    #                                                                  open3d_knn_time * 1000,
    #                                                                  open3d_rnn_time * 1000))
    
if __name__ == '__main__':
    main()