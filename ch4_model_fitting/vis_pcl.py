import open3d as o3d
import os
import struct
import numpy as np

def vis_point_cloud(points):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([point_cloud])

def read_velodyne_bin(path): #, vis = False):
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
    points = np.asarray(pc_list, dtype=np.float64)
    # if vis:
    #     vis_point_cloud(points)
    return points

def vis_sequence_pcl(pcd):
    # The following code achieves the same effect as:
    # o3d.visualization.draw_geometries([pcd])
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()

root_dir = '/home/kit/point_cloud_processing/ch4_model_fitting/datas' # 数据集路径
cat = os.listdir(root_dir)
iteration_num = len(cat)
pcl_list = []
for i in range(iteration_num):
    filename = os.path.join(root_dir, cat[i])
    origin_points = read_velodyne_bin(filename)
    pcd = o3d.geometry.PointCloud()
    import pdb; pdb.set_trace()
    pcd.points = o3d.utility.Vector3dVector(origin_points)
    # pcl_list.append(point_cloud)

for pcd in pcl_list:
    vis_point_cloud(pcd)