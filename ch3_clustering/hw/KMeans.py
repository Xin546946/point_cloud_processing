# 文件功能： 实现 K-Means 算法

import numpy as np
import random

class point():
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __str__(self):
        output = ''
        output += str(self.data) + ' ' + str(self.label)
        return output

def printdata(points):
    for d in points:
        print(d)

def pick_centers(data, k):
    centers = [[] for i in range(k)]
    if data[0].label is -1:
        #random
        centers = random.sample(data,k)
        for i in range(k):
            centers[i] = centers[i].data 
    else:
        for po in data:
            centers[po.label].append(po.data)

        for i in range(k):
            centers[i] = np.mean(centers[i], axis=0)

    return centers



class K_Means(object):
    # k是分组数；tolerance‘中心点误差’；max_iter是迭代次数
    def __init__(self, n_clusters=2, tolerance=0.0001, max_iter=300):
        self.k_ = n_clusters
        self.tolerance_ = tolerance
        self.max_iter_ = max_iter
        self.cluster_centers_ = []
        self.point_with_labels = []

    def fit(self, data):
        # 作业1
        # 屏蔽开始
        iteration = 0
        myData = []
        prev_centers = [[1e10,1e10] for i in range(self.k_)]
        J = 1e10
        for p in data:
            myData.append(point(p,-1))

        while J > self.tolerance_ and iteration < self.max_iter_:
            centers = pick_centers(myData, self.k_)
            J = sum(np.linalg.norm(prev_centers - np.asarray(centers),axis = 1))
            prev_centers = centers
            for samp in myData:
                dist = []
                for c in centers:
                    d = np.linalg.norm(samp.data - c)
                    dist.append(d)
                samp.label = np.argmin(dist)

            iteration += 1
        self.point_with_labels = myData
        self.cluster_centers_ = centers
        # 屏蔽结束

    def predict(self, p_datas):
        result = []
        # 作业2
        # 屏蔽开始
        for data in p_datas:
            result.append(point(data,-1))

        for p in result:
            dist = []
            for c in self.cluster_centers_:
                d = np.linalg.norm(p.data - c)
                dist.append(d)
            p.label = np.argmin(dist)        
        
        for i in range(len(result)):
            result[i] = result[i].label
        # 屏蔽结束
        return result

if __name__ == '__main__':
    x = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])
    k_means = K_Means(n_clusters=2)
    k_means.fit(x)
    printdata(k_means.point_with_labels)

    cat = k_means.predict([[1,1.9]])
    print(cat)
