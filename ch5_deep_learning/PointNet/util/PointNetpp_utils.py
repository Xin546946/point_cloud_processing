import numpy as np

import torch
import torch.nn.functional as F

from time import time

def farthest_point_sample_(xyz, npoint):
    """
    
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    import pdb; pdb.set_trace()

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def farthest_point_sample(points_xyz, num_samples):
    """
    Tutorial: https://blog.csdn.net/dsoftware/article/details/107184116
    Input: 
        point_xyz [batch_size, num_points, channels]
        num_samples: the number of samples of FPS
    Output:
        samples_xyz [batch_size, num_samples, channels] 
    """
    device = points_xyz.device
    batch_size, num_points, channels = points_xyz.shape

    samples_xyz = torch.zeros(batch_size, num_samples, dtype=torch.long).to(device) # samples should have size discussed above 
    distances_container = torch.ones(batch_size, num_points).to(device) * 1e10 # distance matrix for one point and all points(batch_size * num_points)
    
    for i in range(num_samples):
        samples_xyz[:, i] = torch


class SetAbstraction(torch.nn.Module):
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
        
        # if self.group_all

        # 2. group
        
        # 3. PointNet

if __name__ == '__main__':
    points_xyz = torch.randn(5,100,3)
    num_samples = 50
    samples_xyz = farthest_point_sample_(points_xyz, num_samples)