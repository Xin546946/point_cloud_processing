from __future__ import print_function
import torch
import torch.nn.functional as F

import numpy as np

from util.PointNetpp_utils import SetAbstraction

class PointNetpp(nn.Module):

    def __init__(self, num_class: int = 10, normal_channel = True): 
        super(PointNetpp, self).__init__()
        self.num_class = num_class
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel
        self.set_abstraction1 = SetAbstraction(num_points = 512, radius = 0.2, num_sample = 32, 
                                            in_channel = in_channel, mlp = [64, 64, 128], group_all = False)
        self.set_abstraction2 = SetAbstraction(num_points = 128, radius = 0.4, num_sample = 64, 
                                            in_channel = 128 + 3, mlp = [128, 128, 256], group_all = False)
        self.set_abstraction1 = SetAbstraction(num_points = 512, radius = 0.2, num_sample = 32, 
                                            in_channel = 256 + 3, mlp = [256, 512, 1024], group_all = True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, self.num_class)

        def forward(self, points): # points [batch_size, dim, num_points]
            batch_size = points.shape[0]
            if self.normal_channel:
                point_norm = points[:, 3:, :] # get normal vector of points (from the 3nd element of 2nd dim)
                point_xyz = points[:, :3, :] # get point position [x,y,z]
            else:
                point_norm = None
            
            # extract local features: 
            point_xyz, point_features = self.set_abstraction1(point_xyz, point_norm)
            point_xyz, point_feature = self.set_abstraction2(point_xyz, point_features)
            point_xyz, point_features = self.set_abstraction3(point_xyz, point_features)

            # resize features to b * num for mlp
            x = point_features_3.view(batch_size, 1024)

            # mlp with drop out
            x = self.dropout1(F.relu(self.bn1(self.fc1(x))))
            x = self.dropout2(F.relu(self.bn2(self.fc2(x))))
            x = self.fc3(x)

            return x, point_feature_3

