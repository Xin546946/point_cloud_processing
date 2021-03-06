import numpy as np
import random

class Sample(object):
    def __init__(self, data, label = -1):
        self.data = data
        self.label = label

def compute_min_dist_through_datas_and_centers(init_centers_datas, datas):
    if len(init_centers_datas) == 1:
        distances_list = np.linalg.norm(init_centers_datas[0] - datas, axis = 1)
        distances_list_square = np.power(distances_list, 2)
        distances_list_square_normalize = distances_list_square / np.sum(distances_list_square,axis= 0) 
        return distances_list_square_normalize
    else:
        distances_list = []
        for init_center_data in init_centers_datas:
            distances_list.append(np.linalg.norm(np.asarray(init_center_data) - datas, axis = 1))
        min_distances_list = np.min(np.asarray(distances_list), axis = 0)
        min_distances_list_square = np.power(min_distances_list, 2)
        min_distances_list_square_normalize = min_distances_list_square / np.sum(min_distances_list_square,axis= 0) 
        
        return min_distances_list_square_normalize


def init_centers(datas, k, method):
    if method == 'kmeans':
        return init_centers_kmeans(datas, k)
    elif method == 'kmeansplusplus':
        return init_center_plusplus(datas, k)
    else:
        print("Please enter valid method.")

def init_center_plusplus(datas, k):
    center = random.choice(datas)
    
    # np.random.seed(0)
    init_centers_datas = []
    init_centers_datas.append(center)
    
    flag = k-1 
    
    while(flag):
        
        distances = compute_min_dist_through_datas_and_centers(init_centers_datas, datas)
        # print(distances)
        center = np.random.choice(distances, p = distances.ravel())
        index = np.argwhere(distances == center)[0][0]
        init_centers_datas.append(datas[index])
        flag -= 1

    init_centers = [[] for i in range(k)]

    for i in range(k):
        init_centers[i] = Sample(init_centers_datas[i], i)

    return init_centers

def init_centers_kmeans(datas, k):
    init_centers_data = random.sample(list(datas), k)
    init_centers = [[] for i in range(k)]
    for i in range(k):
        init_centers[i] = Sample(init_centers_data[i],i)
    return init_centers

def update_label(samples, centers):
    for sample in samples:
        min_distance = 1e10
        for center in centers:
            distance = np.linalg.norm(center.data - sample.data) 
            if distance < min_distance:
                min_distance = distance
                sample.label = center.label
    return samples

def update_center(centers,samples, k):
    last_centers = []
    for center in centers:
        last_centers.append(Sample(center.data, center.label))
        
    features_per_center = [[] for i in range(k)]
    for sample in samples:
        features_per_center[sample.label].append(sample.data)

    for i in range(k):
        features_per_center_np = np.array(features_per_center[i])
        centers[i].data = np.mean(features_per_center_np, axis = 0)
        # import pdb; pdb.set_trace()
    return last_centers, centers

def compute_distance(last_centers, centers):
    distance = 0.0
    for i in range(len(centers)):
        distance += np.linalg.norm(centers[i].data - last_centers[i].data)
    return distance

if __name__ == '__main__':
    datas = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])
    centers = init_center_plusplus(datas, 3)

    for center in centers:
        print(center.data, center.label)