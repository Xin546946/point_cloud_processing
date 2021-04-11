from __future__ import print_function
import torch
import numpy as np
import torch.nn.functional as F


class PointNetDIY(torch.nn.Module):
    def __init__(self, num_class: int =10, inputDim : int =3, outputDim : int =1024):
        super(PointNetDIY, self).__init__()
        self.num_class_ = num_class
        self.inputDim = inputDim
        self.outputDim = outputDim
        self.dropout = torch.nn.Dropout(p=0.7)
        
        self.conv1 = torch.nn.Conv1d(self.inputDim, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, self.outputDim, 1)
        
        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(self.outputDim)

    def forward(self, x):
        batch_size = x.shape[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, self.outputDim)
        return x
    
class SetAbstraction(torch.nn.Module):
    def __init__(self, num_class, inputDim, outputDim, radius):
        super(SetAbstraction, self).__init__()
        self.num_class = num_class
        self.inputDim = inputDim
        self.outputDim = outputDim
        self.radius = radius
        
    def forward(self, x):
        batch_size = x.shape[0]
        #sample x
        #grouping
        for i in range(batch_size):
            cur_point_cloud = x[i]#?shape??

        model = PointNetDIY(self.num_class, )#init pointnet
        x = model(x)
        return x

class PointNetPP(torch.nn.Module):
    def __init__(self, )