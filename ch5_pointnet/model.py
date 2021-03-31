import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from dataloader import ModelNetDataset

class Pointnet(nn.Module):
    def __init__(self):
        super(Pointnet, self).__init__()
        self.conv1 = nn.Conv1d(3,64,1)
        self.conv2 = nn.Conv1d(64,64,1)
        self.conv3 = nn.Conv1d(64,1024,1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(1024)

    def forward(self, input):
        x = F.relu(self.bn1(self.conv1(input)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)##x is the features
        return x
    
if __name__ == '__main__':
    path = "/home/gfeng/gfeng_ws/modelnet40_dataset"
    mydata = ModelNetDataset(path, split='test')
    ##print(mydata.fns)
    point_cloud, cls = mydata[0]
    print('files loaded')

    mynet = Pointnet()
    y = mynet.forward(point_cloud)
    print(y)