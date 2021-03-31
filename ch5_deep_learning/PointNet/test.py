import os

path = '/mvtec/home/jinx/privat/modelnet40_normal_resampled'

path_train_id = '/mvtec/home/jinx/privat/modelnet40_normal_resampled/modelnet40_train.txt'

with open(path_train_id, 'r') as f:
    for line in f:
        ls = line.strip().split()
        import pdb; pdb.set_trace()