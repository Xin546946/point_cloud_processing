import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

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
    a = np.array([[[1,2,3]],[[2,3,4]]])
    a = torch.from_numpy(a)
    print(a)
    net = Pointnet()
    y = net.forward(a)
    print(y)