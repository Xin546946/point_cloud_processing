import os
import torch
import numpy as np
import torchvision
import random
import tqdm
import copy
import math
import matplotlib.pyplot as plt
from util.utils import read_cloud_from_txt, vis_point_cloud

         
class ModelNetDataset(torch.utils.data.Dataset):
    
    def __init__(self, cloud_folder, data_mode = 'train', data_augmentation = True, num_class_to_use : int=10, normal_channel = False):
        assert data_mode in ['train', 'validation', 'test'] 
        self.cloud_folder_ = cloud_folder
        self.data_augmentation_ = data_augmentation
        self.data_mode_ = data_mode
        self.clouds_labels_ = [] 
        self.num_class_to_use_ = num_class_to_use
        self.data_mode_ = data_mode
        self.labels = None
        self.normal_channel = normal_channel
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
                current_cloud_path = os.path.join(current_cloud_folder, cloud)
                points = read_cloud_from_txt(current_cloud_path)
                if self.normal_channel:
                    self.clouds_labels_.append([points, num_label, current_cloud_path])
                else:
                    self.clouds_labels_.append([points[0:3], num_label, current_cloud_path])
    def __getitem__(self, index):
        

        cloud_label = self.clouds_labels_[index] 
        
        cloud = cloud_label[0] 
        
        # data augmentation
        if self.data_augmentation_:
            theta = np.random.uniform(0, np.pi * 2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
            cloud[:,[0,2]] = cloud[:,[0,2]].dot(rotation_matrix) # random rotation
            cloud += np.random.normal(0, 0.02, size=cloud.shape) # random jitter
        
        return cloud, cloud_label[1], cloud_label[2]
    
    def __len__(self):
        return len(self.clouds_labels_)

    

if __name__ == '__main__':
    #cloud_folder_path = '/mvtec/home/jinx/privat/modelnet40_normal_resampled'
    cloud_folder_path = '/home/gfeng/gfeng_ws/modelnet40_dataset'
    # cloud_folder = torchvision.datasets.ImageFolder(cloud_folder_path)
    
    cloud_dataset_test = ModelNetDataset(cloud_folder=cloud_folder_path, data_augmentation=False ,data_mode = 'test', num_class_to_use=10)
    #cloud_dataset_train = ModelNetDataset(cloud_folder=cloud_folder_path, data_mode = 'train', num_class_to_use=10)
    #cloud_dataset_val = ModelNetDataset(cloud_folder=cloud_folder_path, data_mode = 'validation', num_class_to_use=3)
    cloud_loader = torch.utils.data.DataLoader(cloud_dataset_test, batch_size = 32, shuffle = True, num_workers = 8)

    #print("The length of train: {}, validation: {}, test: {}".format(len(cloud_dataset_train), len(cloud_dataset_val), len(cloud_dataset_test)))
    n = 0
    for point, label in cloud_loader:
        #import pdb; pdb.set_trace()
        n += 1
        cur_poind_cloud = point.transpose(2,1)[0].numpy()
        print(cloud_dataset_test.labels[label[0]])
        vis_point_cloud(cur_poind_cloud)
        if n > 10:
            break
        


        
    