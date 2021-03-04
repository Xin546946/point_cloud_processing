import numpy as np
import random

class Sample(object):
    
    def __init__(self, data, label = -1):
        self.data = data
        self.label = label

datas = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])
samples = [(Sample(data)) for data in datas]
k = 3
for sample in samples:
    print((sample.data, sample.label))
init_centers_data = random.sample(list(datas), k = 3)
init_centers = [[] for i in range(k)]
for i in range(3):
    init_centers[i] = Sample(init_centers_data[i],i)

for sample in samples:
    print(sample.label)


for center in init_centers:
    print(center.data, center.label)
print("---------")

# update label
for sample in samples:
    print("@@@@@@@@@q")
    min_distance = 1e10
    for center in init_centers:
        distance = np.linalg.norm(center.data - sample.data) 
        print("distance of ",sample.data, " and ",center.data ," is ", distance)
        if  distance < min_distance:
            min_distance = distance
            sample.label = center.label
            print(sample.data, "update label to ", sample.label)
            print("Now min_distance is ", min_distance)
    print("Label of sample", sample.data, " is ", sample.label)
##
for sample in samples:
    print(sample.data, sample.label)
print(".-----------------")
# updata center
k = 3

features_per_center = [[] for i in range(k)]
for sample in samples:
    features_per_center[sample.label].append(sample.data)

print(np.asarray(features_per_center[1]))
features_per_center_np = features_per_center[1]
print("------")
print(np.mean(features_per_center_np, axis = 0))

# for i in range(k):
    #  init_centers[i].data = np.mean(