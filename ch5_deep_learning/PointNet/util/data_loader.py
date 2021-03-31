import os
import torch
import numpy as np
import torchvision
import random
import tqdm
import copy
import math
import matplotlib.pyplot as plt

def read_off(filename):
    points = []
    faces = []
    with open(filename, 'r') as f:
        first = f.readline()
        if (len(first) > 4): 
            n, m, c = first[3:].split(' ')[:]
        else:
            n, m, c = f.readline().rstrip().split(' ')[:]
        n = int(n)
        m = int(m)
        for i in range(n):
            value = f.readline().rstrip().split(' ')
            points.append([float(x) for x in value])
        for i in range(m):
            value = f.readline().rstrip().split(' ')
            faces.append([int(x) for x in value])
    points = np.array(points)
    faces = np.array(faces)
    return points, faces


         
def read_cloud_from_txt(cloud_path):
    points_list = [] 
    with open(cloud_path, 'r') as f:
        while True:
            lines = f.readline()
            
            if not lines:
                break
            point = [math.fabs(float(i)) for i in lines.split(',')[0:3]]
            points_list.append(point)
    return np.asarray(points_list, dtype = np.float32).T

class ModelNetDataset(torch.utils.data.Dataset):
    
    def __init__(self, cloud_folder, data_mode = 'train', data_augmentation = True, num_class_to_use : int=10):
        assert data_mode in ['train', 'validation', 'test'] 
        self.cloud_folder_ = cloud_folder
        self.data_augmentation_ = data_augmentation
        self.data_mode_ = data_mode
        self.clouds_labels_ = [] 
        self.num_class_to_use_ = num_class_to_use
        self.data_mode_ = data_mode
        self.labels = None
        self.get_clouds_labels_from_txt_file()
    
    def get_clouds_labels_from_txt_file(self):
        """[get clouds and labels from off file for all object]

        Args:
            cloud_folder_path ([str]): [the file of this object]
            data_mode (str, optional): [choices from train, test]. Defaults to 'train'.
        """
        cat = os.listdir(self.cloud_folder_)
        self.labels = cat
        cat_ = copy.deepcopy(cat)
        for item in cat_:
            if item.endswith('.txt') == True:
                cat.remove(item)
        
                
        for i in range(len(cat)):
            if i == self.num_class_to_use_:
                break
            current_cloud_folder =  os.path.join(self.cloud_folder_, cat[i])
            clouds = os.listdir(current_cloud_folder)
            label = current_cloud_folder.split('/')[-1] 
            num_label = cat.index(label)
            print("Start loading {}....".format(label))
            num_clouds = len(clouds)
            
            for id_, cloud in enumerate(tqdm.tqdm(clouds)):
                if self.data_mode_ == 'train':
                    if id_ < 0.5 * num_clouds:
                        current_cloud_path = os.path.join(current_cloud_folder, cloud)
                        points = read_cloud_from_txt(current_cloud_path)
                        self.clouds_labels_.append([points, num_label])
                elif self.data_mode_ == 'validation':
                    if 0.5 * num_clouds <= id_ < 0.8 * num_clouds:   
                        current_cloud_path = os.path.join(current_cloud_folder, cloud)
                        points = read_cloud_from_txt(current_cloud_path)
                        self.clouds_labels_.append([points, num_label])
                else:
                    if id_ >= 0.8 * num_clouds:    
                        current_cloud_path = os.path.join(current_cloud_folder, cloud)
                        points = read_cloud_from_txt(current_cloud_path)
                        self.clouds_labels_.append([points, num_label])
                    
    def __getitem__(self, index):
        
        # if self.train_or_val_ == 'train':
        #     cloud_label = self.clouds_labels_train_[index] 
        # elif self.train_or_val_ == 'validation':
        #     cloud_label = self.clouds_labels_validation_[index] 
        # else:
        cloud_label = self.clouds_labels_[index] 
        
        cloud = cloud_label[0] 
        
        # data augmentation
        if self.data_augmentation_:
            theta = np.random.uniform(0, np.pi * 2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
            cloud[:,[0,2]] = cloud[:,[0,2]].dot(rotation_matrix) # random rotation
            cloud += np.random.normal(0, 0.02, size=cloud.shape) # random jitter
        
        return cloud, cloud_label[1]   
    
    def __len__(self):
        return len(self.clouds_labels_)
        # if self.train_or_val_ == 'train':
        #     return len(self.clouds_labels_train_)
        # elif self.train_or_val_ == 'validation':
        #     return len(self.clouds_labels_validation_)
        # else:
        #     return len(self.clouds_labels_test_)
    

if __name__ == '__main__':
    cloud_folder_path = '/mvtec/home/jinx/privat/modelnet40_normal_resampled'
    # cloud_folder = torchvision.datasets.ImageFolder(cloud_folder_path)
    
    cloud_dataset_test = ModelNetDataset(cloud_folder=cloud_folder_path, data_mode = 'test', num_class_to_use=2)
    cloud_dataset_train = ModelNetDataset(cloud_folder=cloud_folder_path, data_mode = 'train', num_class_to_use=2)
    cloud_dataset_val = ModelNetDataset(cloud_folder=cloud_folder_path, data_mode = 'validation', num_class_to_use=2)
    cloud_loader = torch.utils.data.DataLoader(cloud_dataset_train, batch_size = 32, shuffle = True, num_workers = 8)
    # import pdb; pdb.set_trace()
    print("The length of train: {}, validation: {}, test: {}".format(len(cloud_dataset_train), len(cloud_dataset_val), len(cloud_dataset_test)))
    for point, label in cloud_loader:
        pass
        
    