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

# def get_clouds_labels_from_off_file(cloud_folder_path, data_mode = 'train'):
#      cat = os.listdir(cloud_folder_path)
#      for i in range(len(cat)):
#         import pdb; pdb.set_trace()
#         current_cloud_files =  os.path.join(cloud_folder_path, cat[i], data_mode)
#         clouds = os.listdir(current_cloud_files)
#         label = current_cloud_files.split('/')[-2] 
#         for id_, cloud in enumerate(clouds):
#             import pdb; pdb.set_trace()
#             points, _ = read_off(os.path.join(current_cloud_files, cloud))
            
#         # points, _ = read_off(current_cloud_file)
         
def read_cloud_from_txt(cloud_path):
    with open(cloud_path, 'r') as f:
        while True:
            lines = f.readline()
            import pdb; pdb.set_trace()
            if not lines:
                break
                pass
            x, y, z = [math.fabs(float(i)) for i in lines.split(',')[0:3]]
    pass

class ModelNetDataset(torch.utils.data.Dataset):
    
    def __init__(self, cloud_folder, train_or_val = None, data_mode = 'train', transform = None, num_class_to_use : int=10):
        assert train_or_val in ['train', 'validation', None] 
        assert data_mode in ['train', 'test'] 
        self.cloud_folder_ = cloud_folder
        self.transform_ = transform
        self.data_mode_ = data_mode
        self.clouds_labels_train_ = [] 
        self.clouds_labels_validation_ = []
        self.clouds_labels_test_ = [] 
        self.num_class_to_use_ = num_class_to_use
        self.train_or_val_ = train_or_val
        
        self.get_clouds_labels_from_off_file()
    
    def get_clouds_labels_from_off_file(self):
        """[get clouds and labels from off file for all object]

        Args:
            cloud_folder_path ([str]): [the file of this object]
            data_mode (str, optional): [choices from train, test]. Defaults to 'train'.
        """
        cat = os.listdir(self.cloud_folder_)
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
            print("Start loading {}....".format(label))
            num_clouds = len(clouds)
            for id_, cloud in enumerate(tqdm.tqdm(clouds)):
                current_cloud_path = os.path.join(current_cloud_folder, cloud)
                points, _ = read_cloud_from_txt(current_cloud_path)
                if id_ < 0.5 * num_clouds:
                    self.clouds_labels_train_.append([points, label])
                elif 0.5 * num_clouds <= id_ < 0.8 * num_clouds:
                    self.clouds_labels_validation_.append([points, label])
                else:
                    self.clouds_labels_test_.append([points, label])
                    
    def __getitem__(self, index):
        
        if self.train_or_val_ == 'train':
            cloud_label = self.clouds_labels_train_[index] 
        elif self.train_or_val_ == 'validation':
            cloud_label = self.clouds_labels_validation_[index] 
        else:
            cloud_label = self.clouds_labels_[index] 
            
        return cloud_label[0], cloud_label[1]   
    
    def __len__(self):
        if self.train_or_val_ == 'train':
            return len(self.clouds_labels_train_)
        elif self.train_or_val_ == 'validation':
            return len(self.clouds_labels_validation_)
        else:
            return len(self.clouds_labels_)
    

if __name__ == '__main__':
    cloud_folder_path = '/mvtec/home/jinx/privat/modelnet40_normal_resampled'
    # cloud_folder = torchvision.datasets.ImageFolder(cloud_folder_path)
    
    cloud_dataset = ModelNetDataset(cloud_folder=cloud_folder_path, data_mode = 'test', num_class_to_use=3)
    cloud_loader = torch.utils.data.DataLoader(cloud_dataset, batch_size = 32, shuffle = True, num_workers = 8)
    import pdb; pdb.set_trace()
    for point, label in cloud_loader:
        pass
        
    