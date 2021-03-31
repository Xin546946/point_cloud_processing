import numpy as np
import json
import os
import sys
import torch

def create_experiment_dir(path):
    # path/train/config.json, train_loss, validation_loss.json
    os.makedirs(path, exist_ok=True)
    os.makedirs(path + '/train', exist_ok=True)
    os.makedirs(path + '/eval', exist_ok=True)
    
    # train_path = os.path.join(path,'train')
    # eval_path = os.path.join(path,'eval')
    # os.system('touch {}/train_loss.json'.format(train_path))
    # os.system('touch {}/validation_loss.json'.format(train_path))
    # os.system('touch {}/config.json'.format(train_path))
    # os.system('touch {}/feature_label.json'.format(eval_path))