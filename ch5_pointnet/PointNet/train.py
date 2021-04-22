import torch
import numpy as np
import torch.nn.functional as F
import tqdm
import matplotlib.pyplot as plt
import os

from util.data_loader import ModelNetDataset
from util.PointNet import PointNet
from util.argparser import parse_arguments
from util.utils import create_experiment_dir

def load_data(cloud_folder_path, data_mode, batch_size, num_class):
    cloud_dataset = ModelNetDataset(cloud_folder=cloud_folder_path, data_mode=data_mode, num_class_to_use=num_class)
    cloud_loader = torch.utils.data.DataLoader(cloud_dataset, batch_size = batch_size, shuffle = True, num_workers = 8)
    return cloud_loader

def run_epoch(data_loader, model, mode, learning_rate=0.0001, weight_decay=1e-5):
    assert mode in ['train', 'validation']
    if mode == 'train':
        model.train()
    else:
        model.eval()
        
    optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate,  weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    loss_list = []
    accuracy_list = []
     
    for i, data in enumerate(tqdm.tqdm(data_loader, 0)):
        point, label, _ = data
        #label = torch.unsqueeze(label, 1)
        batch_size = point.shape[0]
        
        if torch.cuda.is_available():
            point, label = point.cuda(), label.cuda()

        if mode == 'train':
            optimizer.zero_grad()
            predict = model.forward(point)
            loss_ = criterion(predict, label)
            loss_.backward()
            optimizer.step()
        
        else:
            predict = model.forward(point)
            loss_ = criterion(predict, label)

        pred_choice = predict.data.max(1)[1]
        correct = pred_choice.eq(label.data).cpu().sum()
        import pdb; pdb.set_trace()
        loss_list.append(loss_.item())
        accuracy_list.append(correct / batch_size)
        
    loss_return = np.mean(np.asarray(loss_list))
    accuracy_return = np.mean(np.asarray(accuracy_list))
    
    return loss_return, accuracy_return

def print_info(curr_epoch, train_loss_per_epoch, train_accuracy_per_epoch, validation_loss_per_epoch, val_accuracy_per_epoch):
    print(
            "Epoch number {}, Current train loss {}, Current_train_accuracy {}, Current validation loss {}, Current_validation_accuracy {}, ".format(
            curr_epoch, train_loss_per_epoch, train_accuracy_per_epoch, validation_loss_per_epoch, val_accuracy_per_epoch)
            )


def train(args):
    
    create_experiment_dir(args.exp_dir)
    # cloud_folder_path = '/mvtec/home/jinx/privat/modelnet40_normal_resampled'
    
    train_loader = load_data(cloud_folder_path=args.data_dir, data_mode='train',batch_size=args.batch_size, num_class=args.num_class)
    val_loader = load_data(cloud_folder_path=args.data_dir, data_mode='validation',batch_size=args.batch_size, num_class=args.num_class)
    
    print("@@@@@Load model...")
    model = PointNet(num_class=args.num_class)
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    print("Start to train the networks...")
    
    train_loss_history = []
    validation_loss_history = []
    train_accuracy_history = [] 
    val_accuracy_history = [] 
    counter = [] 
    best_validation_loss = float('inf')

    for curr_epoch in range(args.num_epochs):
        counter.append(curr_epoch)
        train_loss_per_epoch, train_accuracy_per_epoch = run_epoch(data_loader = train_loader, model = model, mode='train', learning_rate=args.learning_rate, weight_decay=args.weight_decay)
        validation_loss_per_epoch, val_accuracy_per_epoch = run_epoch(data_loader = train_loader, model = model, mode='validation')

        train_loss_history.append(train_loss_per_epoch)
        validation_loss_history.append(validation_loss_per_epoch)
        
        train_accuracy_history.append(train_accuracy_per_epoch)
        val_accuracy_history.append(val_accuracy_per_epoch)
        
        print_info(curr_epoch, train_loss_per_epoch, train_accuracy_per_epoch, validation_loss_per_epoch, val_accuracy_per_epoch)
        
        if validation_loss_per_epoch < best_validation_loss:
            save_model_path = args.exp_dir + '/train/best_model.pth'
            torch.save(model.state_dict(), save_model_path)
            
        # save and rewrite the train & validation loss for each 10 epoch, save and plot it for each 50 epochs   
        plt.subplots() 
        plt.plot(counter,train_loss_history, label = 'train loss')
        plt.plot(counter,validation_loss_history, label = 'validation loss')
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title('loss diagramm')
        fig_file_name = 'current_loss.png'
        save_path = os.path.join(args.exp_dir + '/train/', fig_file_name)
        plt.savefig(args.exp_dir + '/train/' + fig_file_name)  
        
        plt.subplots() 
        plt.plot(counter,train_accuracy_history, label = 'train accuracy')
        plt.plot(counter,val_accuracy_history, label = 'validation accuracy')
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.title('accuracy diagramm')
        fig_file_name = 'current_accuracy.png'
        save_path = os.path.join(args.exp_dir + '/train/', fig_file_name)
        plt.savefig(args.exp_dir + '/train/' + fig_file_name)  
        
        plt.close('all')
        if curr_epoch % 10 == 0 and curr_epoch != 0:    
            # save model   
            save_model_path = args.exp_dir + '/train/model_at_epoch{}.pth'.format(curr_epoch)
            torch.save(model.state_dict(), save_model_path)
        
        
            
if __name__ == '__main__':
    args = parse_arguments()
    train(args)             