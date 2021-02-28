import numpy as np
import open3d as o3d

pc_list = []
path = '/home/kit/point_cloud_processing/ch2_nearest_neighbor_problem/data'
with open(path, 'rb') as f:
    content = f.read()
    pc_iter = struct.iter_unpack('ffff', content)
    for idx, point in enumerate(pc_iter):
        pc_list.append([point[0], point[1], point[2]])
points = np.asarray(pc_list, dtype=np.float32)
