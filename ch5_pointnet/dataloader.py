import torch.utils.data as data
import os
import numpy as np
import torch 


class ModelNetDataset(data.Dataset):
    def __init__(self, path, data_agumentation=True):
        self.root = path
        self.data_agumentation = data_agumentation
        self.fns = []
        with open(os.path.join(self.root, '{}.txt'.format('filelist')), 'r') as f:
            for line in f:
                self.fns.append(line.strip())
        
        self.cat = {}##use hashtable to store catagories and their indices
        with open(os.path.join(self.root, '{}.txt'.format('modelnet40_shape_names')), 'r') as f:
            i = 0
            for line in f:
                ls = line.strip()
                self.cat[ls] = int(i)
                i += 1
        #print(self.cat)
        self.classes = list(self.cat.keys())

    def __getitem__(self, index):
        fn = self.fns[index]
        cata = self.cat[fn.split('/')[0]]
        point_cloud = []
        with open(os.path.join(self.root, fn), 'r') as f:
            for line in f:
                point = []
                for i in range(3):
                    point.append(np.float32(line.strip().split(',')[i]))
                point_cloud.append(point)
        point_cloud = np.asarray(point_cloud)
        point_set = torch.from_numpy(point_cloud.astype(np.float32))
        cata = torch.from_numpy(np.array([cata]).astype(np.int64))
        return point_set, cata

    def __len__(self):
        return len(self.root)

if __name__ == '__main__':
    path = "/home/gfeng/gfeng_ws/modelnet40_dataset"
    mydata = ModelNetDataset(path)
    ##print(mydata.fns)
    point_cloud, cls = mydata[0]
    print('files loaded')
