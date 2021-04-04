from __future__ import print_function
import torch
import numpy as np
import torch.nn.functional as F

class PointNet(torch.nn.Module):
    def __init__(self, num_class: int =10):
        super(PointNet, self).__init__()
        self.num_class_ = num_class
        self.dropout = torch.nn.Dropout(p=0.7)
        
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        #self.conv2 = torch.nn.Conv1d(64, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        
        self.fc1 = torch.nn.Linear(1024, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, self.num_class_)
        
        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(1024)
        self.bn4 = torch.nn.BatchNorm1d(512)
        self.bn5 = torch.nn.BatchNorm1d(256)
        
        self.relu = torch.nn.ReLU(inplace=True)
        
    def forward(self, x):
        #import pdb; pdb.set_trace()
        batch_size = x.shape[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn1(self.conv2(x)))
        x = F.relu(self.bn1(self.conv2(x)))

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
    
        
        x = F.relu(self.bn4(self.dropout(self.fc1(x))))
        x = F.relu(self.bn5(self.dropout(self.fc2(x))))
        x = self.fc3(x)

        return x
    
if __name__ == '__main__':
    dummy_input = torch.rand(32,3,2500)
    net = PointNet(num_class=10)
    output = net(dummy_input)