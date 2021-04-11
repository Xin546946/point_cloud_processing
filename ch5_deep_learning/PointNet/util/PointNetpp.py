from __future__ import print_function
import sys
sys.path.append("/home/gfeng/gfeng_ws/point_cloud_processing/ch5_deep_learning/PointNet")

import torch
import torch.nn.functional as F

import numpy as np

from util.PointNetpp_utils import SetAbstraction

class PointNetpp(torch.nn.Module):

    def __init__(self, num_class: int = 10, normal_channel = True): 
        super(PointNetpp, self).__init__()
        self.num_class = num_class
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel
        self.sa1 = SetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128], group_all=False)
        self.sa2 = SetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = SetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        self.fc1 = torch.nn.Linear(1024, 512)
        self.bn1 = torch.nn.BatchNorm1d(512)
        self.dropout1 = torch.nn.Dropout(0.4)
        self.fc2 = torch.nn.Linear(512, 256)
        self.bn2 = torch.nn.BatchNorm1d(256)
        self.dropout2 = torch.nn.Dropout(0.4)
        self.fc3 = torch.nn.Linear(256, self.num_class)

    def forward(self, xyz): # points [batch_size, dim, num_points]
        # extract local features: 
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None

        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        _, l3_points = self.sa3(l2_xyz, l2_points)
        

        # resize features to b * num for mlp
        x = l3_points.view(B, 1024)

        # mlp with drop out
        x = self.dropout1(F.relu(self.bn1(self.fc1(x))))
        x = self.dropout2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)

        return x, l3_points

if __name__ == '__main__':
    model = PointNetpp(1)
    dummy_input = torch.rand(10,6,100)
    dummy_position, dummy_feature = model(dummy_input)

