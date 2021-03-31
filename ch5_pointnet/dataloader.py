import torch.utils.data as data
import os
import numpy as np
import torch 


class ModelNetDataset(data.Dataset):
    def __init__(self, path, data_augumentation=True, split = 'train'):
        self.root = path
        self.data_augumentation = data_augumentation
        self.split = split
        self.fns = []
        self.cat = {}##use hashtable to store catagories and their indices
        with open(os.path.join(self.root, '{}.txt'.format('modelnet40_shape_names')), 'r') as f:
            i = 0
            for line in f:
                ls = line.strip()
                self.cat[ls] = int(i)
                i += 1

        with open(os.path.join(self.root, 'modelnet40_{}.txt'.format(split)), 'r') as f:
            for line in f:
                self.fns.append('{}.txt'.format(line.strip()))
        #print(self.cat)
        self.classes = list(self.cat.keys())

    def __getitem__(self, index):
        fn = self.fns[index]
        cat_name = ''
        name_parts = fn.split('_')
        if len(name_parts) != 2:
            cat_name = name_parts[0] + '_' + name_parts[1]
        else:
            cat_name = name_parts[0]
        label = self.cat[cat_name]
        point_cloud = []
        with open(os.path.join(self.root, cat_name, fn), 'r') as f:
            for line in f:
                point = []
                for i in range(3):
                    point.append(np.float32(line.strip().split(',')[i]))
                point_cloud.append(point)
        point_cloud = np.asarray(point_cloud)

        if self.data_augumentation:
            theta = np.random.uniform(0, np.pi * 2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
            point_cloud[:,[0,2]] = point_cloud[:,[0,2]].dot(rotation_matrix) # random rotation
            point_cloud += np.random.normal(0, 0.02, size=point_cloud.shape) # random jitter
        
        point_set = torch.from_numpy(point_cloud.astype(np.float32))
        label = torch.from_numpy(np.array([label]).astype(np.int64))
        return point_set, label

    def __len__(self):
        return len(self.fns)

if __name__ == '__main__':
    path = "/home/gfeng/gfeng_ws/modelnet40_dataset"
    training_dataset = ModelNetDataset(path, data_augumentation=True, split = 'train')
    test_dataset = ModelNetDataset(path, data_augumentation=False, split = 'test')
    training_data_loader = torch.utils.data.DataLoader(training_dataset, 
                                                        batch_size=8, 
                                                        shuffle=True, 
                                                        num_workers=0)
    testing_data_loader = torch.utils.data.DataLoader(test_dataset, 
                                                      batch_size=8, 
                                                      shuffle=False, 
                                                      num_workers=0)
  
    for i,data in enumerate(training_data_loader,0):
        point,label = data
        if i<10:
            print(label)
    
    print("size of training data: {}, testing data : {}".format(len(training_dataset),len(test_dataset)))