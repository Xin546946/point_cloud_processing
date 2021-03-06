# 文件功能： 实现 K-Means 算法

import numpy as np
import random
from KMeans_components import init_centers, update_label, update_center, Sample, compute_distance



class K_Means(object):
    # k是分组数；tolerance‘中心点误差’；max_iter是迭代次数
    def __init__(self, method = 'kmeansplusplus', n_clusters=2, tolerance=0.00001, max_iter=300):
        self.k_ = n_clusters
        self.tolerance_ = tolerance
        self.max_iter_ = max_iter
        self.centers = None
        self.samples = None
        self.method = method

    def fit(self, datas):
        # 作业1
        # 屏蔽开始
        self.samples = [(Sample(data)) for data in datas]
        self.centers = init_centers(datas, self.k_, self.method)
        tolerance = 1e10
        iteration = 0
        while(tolerance > self.tolerance_ and iteration < self.max_iter_):
            
            iteration += 1
            self.samples = update_label(self.samples, self.centers)
            
            last_centers,centers = update_center(self.centers,self.samples,self.k_)
            tolerance = compute_distance(last_centers, centers)
            print("Iteration : {}, Tolerance : {}".format(iteration, tolerance))
            
        return self.samples, self.centers
        # 屏蔽结束

    def predict(self, p_datas):
        # result = []
        # 作业2
        # 屏蔽开始
        p_samples = [(Sample(p_data)) for p_data in p_datas]

        result = update_label(p_samples, self.centers)
        # import pdb; pdb.set_trace()
        result_labels = []
        for sample in result:
            result_labels.append(sample.label)
        # 屏蔽结束
        return result_labels

if __name__ == '__main__':
    x = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])
    y = np.array([[0.9,2.1], [8.8,10.1]])
    k_means = K_Means(n_clusters=2)
    k_means.fit(x)
    print("Fit finished")
    cat = k_means.predict(y)
    # for c in cat:
    #     print(c.data, c.label)
    print(cat)

