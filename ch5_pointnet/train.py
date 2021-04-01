import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
import random
import torch
import torch.optim as optim
import torch.utils.data
from dataloader import ModelNetDataset
from model import Pointnet
import torch.nn.functional as F
import torch.nn as nn

def load_data(path, data_augumentation, split, batch_size, shuffle, num_workers):
    dataset = ModelNetDataset(path, data_augumentation=data_augumentation, split=split)
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers)
    return data_loader

def train(numOfEpochs):
    
    path = "/home/gfeng/gfeng_ws/modelnet40_dataset"
    training_dataset = load_data(path, True, 'train', 16, True, 4)
    print("Data loaded")

    model = Pointnet()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001, weight_decay=1e-5)
    #loss_fn = torch.nn.CrossEntropyLoss()
    print("Model loaded, meta parameters set")
    if torch.cuda.is_available():
        model = model.cuda()

    print("Training PointNet")
    training_loss = []
    training_acc = []

    for epoch in range(numOfEpochs):
        for i,data in enumerate(training_dataset,0):
            points, target = data
            target = target[:, 0]##16

            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()
            optimizer.zero_grad()
            pred = model.forward(points)##16,40

            loss = F.nll_loss(pred, target)
            loss.backward()
            optimizer.step()

            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
        print("epoch",epoch, ", current loss is :", loss.item(), " accuracy is :", correct/16)

if __name__ == '__main__':
    train(numOfEpochs=5)
