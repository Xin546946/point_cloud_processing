# 实现voxel滤波，并加载数据集中的文件进行验证

import open3d as o3d 
import os
import numpy as np
import math
from pyntcloud import PyntCloud

# 功能：对点云进行voxel滤波
# 输入：
#     point_cloud：输入点云
#     leaf_size: voxel尺寸
def hash_conflict(a,b):
    if(a[1] != b[1] or a[2] != b[2] or a[3] != b[3]):
        return True
    return False


def voxel_filter(point_cloud, leaf_size, random_sampling = False):

    filtered_points = []
    # 作业3
    # 屏蔽开始
    point_cloud = np.asarray(point_cloud)
    print("total points : ", len(point_cloud))
    x_max, y_max, z_max = np.max(point_cloud,axis = 0)
    x_min, y_min, z_min = np.min(point_cloud,axis = 0)

    Dx = (x_max - x_min) // leaf_size
    Dy = (y_max - y_min) // leaf_size
    Dz = (z_max - z_min) // leaf_size
    container_size = Dx * Dy * Dz 

    h = list() 
    for i in range (point_cloud.shape[0]):
        x, y, z = point_cloud[i]
        hx = np.floor((x - x_min) / leaf_size)
        hy = np.floor((y - y_min) / leaf_size)
        hz = np.floor((z - z_min) / leaf_size)
        h.append([(hx + hy * Dx + hz * Dx * Dy) % container_size, hx, hy, hz, i])#why %size?????
    h = np.asarray(h)
    h_index = np.lexsort((h[:,0], h[:,1], h[:,2], h[:,3]))
    H = list()
    for i in range(len(h_index)):
        H.append(h[h_index[i]])
    #now H stores all points, they are all sorted according to different dimensions(h,hx,hy,hz)
 
    filtered_points = list()
    cur_voxel = list()
    cur_voxel.append(point_cloud[int(H[0][4])])#first point can't be conflicted by definition. also avoiding empty cur_voxel

    for i in range(1 , len(h)):
        if (H[i][0] == H[i-1][0] and not hash_conflict(H[i],H[i-1])):
            #put point if it is not conflicted and have same voxel index.
            cur_voxel.append(point_cloud[int(H[i][4])])
        else:
            #otherwise pick sample point from currect voxel then clear it and add this point into it 
            if(random_sampling == False):
                [x_c, y_c, z_c] = np.mean(np.asarray(cur_voxel),axis = 0)
            else:
                index = np.random.choice(np.asarray(cur_voxel).shape[0], 1)
                [x_c, y_c, z_c] = cur_voxel[index[0]]
            filtered_points.append([x_c,y_c,z_c])
            cur_voxel.clear()
            cur_voxel.append(point_cloud[int(H[i][4])])
    # 屏蔽结束



    # 把点云格式改成array，并对外返回
    filtered_points = np.array(filtered_points, dtype=np.float64)
    print("sample points : ",len(filtered_points))
    return filtered_points

def main():
    # # 从ModelNet数据集文件夹中自动索引路径，加载点云
    # cat_index = 10 # 物体编号，范围是0-39，即对应数据集中40个物体
    # root_dir = '/Users/renqian/cloud_lesson/ModelNet40/ply_data_points' # 数据集路径
    # cat = os.listdir(root_dir)
    # filename = os.path.join(root_dir, cat[cat_index],'train', cat[cat_index]+'_0001.ply') # 默认使用第一个点云
    # point_cloud_pynt = PyntCloud.from_file(file_name)

    # 加载自己的点云文件
    file_name = "/home/gfeng/gfeng_ws/point_cloud_processing/ch1_introduction/hw1/ply_data/airplane/test/1.ply"
    point_cloud_pynt = PyntCloud.from_file(file_name)

    # 转成open3d能识别的格式
    point_cloud_o3d = point_cloud_pynt.to_instance("open3d", mesh=False)
    # o3d.visualization.draw_geometries([point_cloud_o3d]) # 显示原始点云

    # 调用voxel滤波函数，实现滤波
    filtered_cloud = voxel_filter(point_cloud_pynt.points, 100.0, False)
    point_cloud_o3d.points = o3d.utility.Vector3dVector(filtered_cloud)
    # 显示滤波后的点云
    o3d.visualization.draw_geometries([point_cloud_o3d])

if __name__ == '__main__':
    main()
