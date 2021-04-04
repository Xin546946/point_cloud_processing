import open3d as o3d
from data_loader import read_cloud_from_txt
from utils import vis_point_cloud

if __name__ == '__main__':
    path = '/home/gfeng/gfeng_ws/modelnet40_dataset/airplane/airplane_0001.txt'
    point_cloud = read_cloud_from_txt(path)
    vis_point_cloud(point_cloud.T)
