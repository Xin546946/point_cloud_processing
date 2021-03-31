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
