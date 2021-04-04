import torch
import numpy as np
import torch.nn.functional as F
import tqdm
import matplotlib.pyplot as plt
import os
from torch.autograd import Variable
from util.data_loader import ModelNetDataset
from util.model import PointNet
from util.argparser import parse_arguments

def load_data(cloud_folder_path, data_mode, batch_size, num_class):
    cloud_dataset = ModelNetDataset(cloud_folder=cloud_folder_path, data_mode=data_mode, num_class_to_use=num_class)
    cloud_loader = torch.utils.data.DataLoader(cloud_dataset, batch_size = batch_size, shuffle = True, num_workers = 8)
    return cloud_loader

def evaluate(args):
    test_loader = load_data(cloud_folder_path=args.data_dir, data_mode='test',batch_size=args.batch_size, num_class=args.num_class)

    print("@@@@@Load model...")
    model = PointNet(num_class=args.num_class)

    if torch.cuda.is_available():
        model = model.cuda()

    print("Start to evaluate the networks...")
    model_path = os.path.join(args.exp_dir ,'train/best_model.pth')
    model.load_state_dict(torch.load(model_path))
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()

    accuracy_list = []

    for i, data in enumerate(test_loader, 0):
        points, target = data
        batch_size = points.shape[0]
        if torch.cuda.is_available():
            points, target = points.cuda(), target.cuda()
        
        pred = model(points)
        loss = criterion(pred, target)

        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        acc = correct / batch_size
        accuracy_list.append(acc)
    
    print('Average accuracy of the model is : ', np.mean(np.asarray(accuracy_list), axis = 0))

if __name__ == '__main__':
    args = parse_arguments()
    evaluate(args)