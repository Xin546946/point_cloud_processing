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
def clustering(data):
    # 作业2
    # 屏蔽开始
    ##DBSCAN
    pcd_seg = o3d.geometry.PointCloud()
    pcd_seg.points = o3d.utility.Vector3dVector(data)   
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        clusters_index = np.array(
            pcd_seg.cluster_dbscan(eps=0.5, min_points=10, print_progress=True))
    import pdb; pdb.set_trace()
    max_label = clusters_index.max()
    print(f"point cloud has {max_label + 1} clusters")
    colors = plt.get_cmap("tab20")(clusters_index / (max_label if max_label > 0 else 1))
    colors[clusters_index < 0] = 0
    pcd_seg.colors = o3d.utility.Vector3dVector(colors[:, :3])
        
    return pcd_seg