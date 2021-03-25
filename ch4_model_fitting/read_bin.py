import numpy as np
import os
import struct
import math
import copy
import random
from sklearn import cluster, datasets, mixture
from sklearn.preprocessing import normalize
from itertools import cycle, islice
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d

path = 'datas/007515.bin'
pc_list = []
with open(path, 'rb') as f:
    content = f.read()
    pc_iter = struct.iter_unpack('ffff', content)
    for _, point in enumerate(pc_iter):
        import pdb; pdb.set_trace()
        pc_list.append([point[0], point[1], point[2]])
points = np.asarray(pc_list, dtype=np.float32)
print(points)
