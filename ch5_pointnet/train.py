import os
import random
import torch
import torch.optim as optim
import torch.utils.data
from dataloader import ModelNetDataset
from model import Pointnet
import torch.nn.functional as F
import torch.nn as nn

lr = 0.00001
def getModel():
    model = Pointnet()
    opt = torch.optim.Adam(model.parameters(), lr = lr) ##.parametes returns trainable parameters
    return model, opt

if __name__ == '__main__':
    loss_fn = nn.BCELoss()   
    model, opt = getModel()
    batch_size = 10

    train_dataset = ModelNetDataset(path = "/home/gfeng/gfeng_ws/modelnet40_dataset", split = 'train')
    test_dataset = ModelNetDataset(path = "/home/gfeng/gfeng_ws/modelnet40_dataset", split = 'test')
    train_data_loader = torch.utils.data.DataLoader(train_dataset, 
                                                    batch_size=batch_size,
                                                    shuffle = True,
                                                    num_workers=int(opt.workers))
    test_data_loader = torch.utils.data.DataLoader(test_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=False
                                                   num_workers=int(opt.workers))