# 实现voxel滤波，并加载数据集中的文件进行验证

import open3d as o3d 
import os
import numpy as np
from pyntcloud import PyntCloud

# 功能：对点云进行voxel滤波
# 输入：
#     point_cloud：输入点云
#     leaf_size: voxel尺寸
def voxel_filter(point_cloud, leaf_size):
    filtered_points = []
    # 作业3
    # 屏蔽开始
    # compute the min or max of the point set {p1,p2,p3,...}
    x_max, y_max, z_max = np.max(point_cloud,axis = 0)
    x_min, y_min, z_min = np.min(point_cloud,axis = 0)

    # compute the dim of the voxel grid
    Dx = np.round((x_max - x_min) / leaf_size) 
    Dy = np.round((y_max - y_min) / leaf_size)
    Dz = np.round((z_max - z_min) / leaf_size)
    
    # compute voxel idx for each point
    h = list()
    
    for i in range(len(point_cloud)):
        x,y,z = point_cloud[i]
        hx = np.floor((x - x_min) / r)
        hy = np.floor((y - y_min) / r)
        hz = np.floor((z - z_min) / r) 
        h.append(hx + hy * Dx + hz * Dx * Dy)
    h = np.asarray(h)

    h_sorted_idx = np.argsort(h)
    h_sorted_pts = h[h_sorted_idx]

    begin = 0
    for i in range(len(h_sorted_idx) -1):
        if h_sorted_idx[i] == h_sorted_idx[i + 1]:
            continue
        else:
            point_idx = h_sorted_idx[begin, i + 1]
            filtered_points.append(np.mean(point_cloud[point_idx], axis = 0))
            begin = i

    # 屏蔽结束

    # 把点云格式改成array，并对外返回
    filtered_points = np.array(filtered_points, dtype=np.float64)
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
    # o3d.visualization.draw_geometries([point_cloud_o3d]) # 显示原始点云

    # 调用voxel滤波函数，实现滤波
    filtered_cloud = voxel_filter(point_cloud_pynt.points, 100.0)
    point_cloud_o3d.points = o3d.utility.Vector3dVector(filtered_cloud)
    # 显示滤波后的点云
    o3d.visualization.draw_geometries([point_cloud_o3d])

if __name__ == '__main__':
    main()
