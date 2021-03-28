import numpy as np
import os
import struct
import math
import copy
import random
from sklearn import cluster, datasets, mixture
from sklean.neighbors import KDTree
from sklearn.preprocessing import normalize
from itertools import cycle, islice
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d
from test import read_velodyne_bin

class Sample(object):
    def __init__(self,data):
        self.data = data
        self.visited = False
        self.label = -1
        self.mode = None # ['core', 'noise', 'border']

class DBSCAN(object):
    def __init__(self, min_samples, radius):
        self.min_samples = min_samples
        self.radius = radius
        self.leaf_size = leaf_size
        self.samples = []
        self.cluster_counter = 0
        self.tree = None
        self.data = data

    def build_samples(self, data):
        for d in data:
            self.samples.append(Sample(d))

    def build_kdtree(self, data):
        self.kdtree = KDTree(data, leaf_size=self.leaf_size)

    def find_neighbors(self, sample):
        if sample.mode == 'core':
            sample.label = self.cluster_counter
            core_point_neighbour_indices = self.tree.query_radius(self.data[i], r = self.radius)
            for i in core_point_neighbour_indices:
                self.find_neighbours(self.samples[i])


    def fit(self, data):
        self.data = data
        self.build_samples(data)
        # while(not self.all_points_are_visited):
        for sample in self.samples:
            if sample.visited == True:
                continue
            
            sample.visited = True
            indices = self.tree.query_radius(sample.data, r=self.radius)
            if len(indices) <= self.min_samples:
                sample.mode = 'noise'
                continue
        
            # sample.mode = 'core'
            sample.label = self.cluster_counter
            while True:
                neighbour_ids_update =set()
                for i in indices:
                    if self.samples[i].visited == False:
                        continue
                    core_point_neighbour_indices = self.tree.query_radius(self.samples[i].data, r=self.radius)
                    if len(core_point_neighbour_indices) < self.min_samples:
                        self.samples[i].mode = 'border'
                        continue
                    self.samples[i].mode = 'core'
                    self.samples[i].label = self.cluster_counter
                    for id_ in core_point_neighbour_indices:
                        if self.samples[id_].visited == False:
                    # self.samples[i].label = self.cluster_counter
                            neighbour_ids_update.add(id_)
                neighbour_ids = np.array([i for i in neighbour_ids_update if self.samples[i].visited == False])
                if len(neighbour_ids) == 0:
                    break
                neighbour_points



# 功能：从点云中提取聚类
# 输入：
#     data: 点云（滤除地面后的点云）
# 输出：
#     clusters_index： 一维数组，存储的是点云中每个点所属的聚类编号（参考上一章内容容易理解）
def clustering(data):
    # 作业2
    # 屏蔽开始


    # 屏蔽结束

    return clusters_index

def main():
    n_samples = 1500
    noisy_circles = datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05)
    datas, labels = noisy_circles
    
    dbscan = DBSCAN(min_samples, radius)
    dbscan.fit(datas)
    print(dbscan.labels)

    # colors = ['b','r']
    # plt.figure(figsize=(7,7))
    # # import pdb; pdb.set_trace()
    # for i in range(datas.shape[0]):
    #     plt.scatter(datas[i][0], datas[i][1], c = colors[labels[i]], s=2)

    # plt.show()

if __name__ == '__main__':
    main()