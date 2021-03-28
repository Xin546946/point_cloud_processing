import numpy as np
import pylab
import random
import math
import copy
from sklearn import datasets
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.spatial import KDTree
from scipy.stats import multivariate_normal
import tqdm
plt.style.use('seaborn')


class DBSCAN(object):
    def __init__(self, r=0.15, min_samples=5):
        self.r = r
        self.min_samples = min_samples
        self.n_cluster = 0
        self.labels = None

    def fit(self, X):
        N = len(X)
        tags = np.array([0] * N)  # 0-NoVisited, 1-CorePoint, 2-BorderPoint, 3-outlier
        kdtree = KDTree(X, leafsize=8)

        self.labels = [-1] * N   # element: -1: outliers
        for i in tqdm.tqdm(range(N)):
            # import pdb; pdb.set_trace()
            if tags[i]:  # continue if not visited
                continue

            # get current core point's neighbor
            neighbor_ids = kdtree.query_ball_point(X[i, :], r=self.r)
            if len(neighbor_ids) < self.min_samples:
                tags[i] = 3  # outliers
                continue

            tags[i] = 1  # core point
            self.labels[i] = self.n_cluster  # current class id
            neighbor_points = X[neighbor_ids]
            # import pdb; pdb.set_trace()
            while True:
                # update core point and refind its neighbor as the same class, and repeat
                neighbor_ids_update = set()
                for ii, (index, pp) in enumerate(zip(neighbor_ids, neighbor_points)):
                    if tags[index]:
                        continue
                    ng_ids = kdtree.query_ball_point(pp, r=self.r)
                    if len(ng_ids) < self.min_samples:
                        tags[index] = 2    # border point
                        continue
                    tags[index] = 1
                    self.labels[index] = self.n_cluster
                    for id_ in ng_ids:
                        if not tags[id_]:   # No Visited
                            neighbor_ids_update.add(id_)
                # 从neighbor中剔除tag不为0的元素
                neighbor_ids = np.array([i for i in neighbor_ids_update if tags[i] == 0])
                if len(neighbor_ids) == 0:
                    break
                neighbor_points = X[neighbor_ids]
            self.n_cluster += 1  # next class

        print('class: ', self.n_cluster)
        self.labels = np.asarray(self.labels, dtype=np.int32)

if __name__ == '__main__':

    X = np.array([[1.0,1.0],[1.3,1.2],[0.9,1.1], [5,3],[5,1],[10,10],[10.1,10.1],[10.5,10.5]])

    # X, label_gt = datasets.make_circles(n_samples=500, factor=0.5, noise=0.05)

    dbscan = DBSCAN()
    dbscan.fit(X)