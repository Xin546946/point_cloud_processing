# 实现voxel滤波，并加载数据集中的文件进行验证

import open3d as o3d 
import os
import numpy as np
import pandas as pd
from pyntcloud import PyntCloud
from functools import cmp_to_key


def hash_table_conflict(h_lhs, h_rhs):
    if h_lhs[0] == h_rhs[0] and h_lhs[1] == h_rhs[1] and h_lhs[2] == h_rhs[2]:
        return False
    else:
        return True


# 功能：对点云进行voxel滤波
# 输入：
#     point_cloud：输入点云
#     leaf_size: voxel尺寸
def voxel_filter(point_cloud, leaf_size, random_downsampling = False):
    point_cloud = np.asarray(point_cloud)
    print("Number of point clouds is", len(point_cloud))
    filtered_points = []
    # 作业3
    # 屏蔽开始
    # compute the min or max of the point set {p1,p2,p3,...}
    x_max, y_max, z_max = np.max(point_cloud,axis = 0)
    x_min, y_min, z_min = np.min(point_cloud,axis = 0)

    # compute the dim of the voxel grid    
    Dx = (x_max - x_min) // leaf_size
    Dy = (y_max - y_min) // leaf_size
    Dz = (z_max - z_min) // leaf_size
    container_size = Dx * Dy * Dz
    # compute voxel idx for each point
    h = list()
    
    for i in range(len(point_cloud)):
        x,y,z = point_cloud[i]
        hx = np.floor((x - x_min) / leaf_size)
        hy = np.floor((y - y_min) / leaf_size)
        hz = np.floor((z - z_min) / leaf_size)
        hash_table = int(hx + hy * Dx + hz * Dx * Dy) % container_size
        h.append(np.asarray([hx, hy, hz, hash_table]))
    # h_sorted = np.asarray(sorted(h, key = cmp_to_key(lambda lhs, rhs : lhs[3] - rhs[3]))) # not solve hash conflict
    
    h = np.asarray(h)
    h_sorted_idx =np.lexsort((h[:,-1], h[:,0], h[:,1])) # sort h according to h, hx, and hy
    h_sorted = h[h_sorted_idx] # sort h according to new idx
    
    current_voxel = list()
    current_h = list()
    current_voxel.append(point_cloud[0])
    for i in range(1, point_cloud.shape[0]):
        # if no hash conflict
        if h_sorted[i -1, -1] == h_sorted[i, -1] and not hash_table_conflict(h_sorted[i -1,0:3], h_sorted[i, 0:3]):
            current_voxel.append(point_cloud[h_sorted_idx[i]]) # put points in the current_voxel
        else:
            if(random_downsampling == False):
                filtered_points.append(np.mean(np.array(current_voxel), axis = 0))
            else:
                random_idx = np.random.randint(len(current_voxel), size = 1)
                filtered_points.append(current_voxel[int(random_idx)])
            current_voxel.clear() # clear current_voxel for next filtered_point
            current_voxel.append(point_cloud[i])
    # 屏蔽结束

    # 把点云格式改成array，并对外返回
    filtered_points = np.array(filtered_points, dtype=np.float64)
    print("Number of filtered points", len(filtered_points))
    return filtered_points

def main():
    # # 从ModelNet数据集文件夹中自动索引路径，加载点云
    # cat_index = 10 # 物体编号，范围是0-39，即对应数据集中40个物体
    # root_dir = '/Users/renqian/cloud_lesson/ModelNet40/ply_data_points' # 数据集路径
    # cat = os.listdir(root_dir)
    # filename = os.path.join(root_dir, cat[cat_index],'train', cat[cat_index]+'_0001.ply') # 默认使用第一个点云
    # point_cloud_pynt = PyntCloud.from_file(file_name)

    # 加载自己的点云文件
    file_name = "/home/kit/point_cloud_processing/ch1_introduction/hw1/ply_data/airplane/test/1.ply"
    point_cloud_pynt = PyntCloud.from_file(file_name)

    # 转成open3d能识别的格式
    point_cloud_o3d = point_cloud_pynt.to_instance("open3d", mesh=False)
    # o3d.visualization.draw_geometries([point_cloud_o3d], "Orignal point cloud") # 显示原始点云

    # 调用voxel滤波函数，实现滤波

    filtered_cloud = voxel_filter(point_cloud_pynt.points,100.0, True)
    point_cloud_o3d.points = o3d.utility.Vector3dVector(filtered_cloud)
    # 显示滤波后的点云
    o3d.visualization.draw_geometries([point_cloud_o3d],"Voxel grid downsampling")

if __name__ == '__main__':
    main()
