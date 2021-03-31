import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.autograd import Variable
from dataloader import ModelNetDataset

class Pointnet(nn.Module):
    def __init__(self):
        super(Pointnet, self).__init__()
        self.conv1 = nn.Conv1d(3,64,1)
        self.conv2 = nn.Conv1d(64,64,1)
        self.conv3 = nn.Conv1d(64,1024,1)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)##change this to num of classes

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, input):
        batchsize = input.size()[0]
        x = F.relu(self.bn1(self.conv1(input)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)##x is the features

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        return x

if __name__ == '__main__':
    sim_data = Variable(torch.rand(32,3,2500))
    model = Pointnet()
    loss_fn = nn.CrossEntropyLoss() 
    pred = model(sim_data)
    print('prediction', pred)
    print('loss')
