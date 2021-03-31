import torch.utils.data as data
import os
import numpy as np
import torch 


class ModelNetDataset(data.Dataset):
    def __init__(self, path, data_argumentation=True):
        self.root = path
        self.data_argumentation = data_argumentation


    def __getitem__(self, index):
        return self.root[index]

    def __len__(self):
        return len(self.root)


