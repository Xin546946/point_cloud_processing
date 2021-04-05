import numpy as np

import torch
import torch.nn.functional as F

from time import time

# class PointNetModule(nn.Module):
#     def __init_(self, points, )


class SetAbstraction(nn.Module):
    # 1. FPS sampling
    # 2. group
    # 3. PointNet
    def __init__(self, num_points, radius, num_samples, in_channel, mlp, group_all):
        super(SetAbstraction, self).__init__()
        self.num_points = num_points
        self.radius = radius
        self.num_samples = num_samples
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()

        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        
        self.group_all = group_all # TODO what is the group all ???????
    
    def forward(self, point_xyz, point_features):
        """
        Input:
            point_xyz: input points feature in euclidean space [batch_size, channel(3), num_points]
            point_features: input points feature: [batch_size, dim, num_points] # except x,y,z posiiton
        Output:
            new_xyz: resampled points feature in euclidean space [batch_size, channel(3), num_points_resampled]
            new_points: points features: [batch_size, dim', num_points_resampled]
        """
        # 1. sampling
        point_xyz = point_xyz.permute(0, 2, 1)
        if point_features is not None: # point features is normal vector in this case
            point_features = points.permute(0, 2, 1)
        
        if self.group_all

        # 2. group
        
        # 3. PointNet
        