import numpy as np
import random

class Sample(object):
    def __init__(self, data, label = -1):
        self.data = data
        self.label = label


def init_center(datas, k):
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
    last_centers = centers
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