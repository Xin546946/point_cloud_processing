import numpy as np
import json
import os
import sys
import torch
import open3d as o3d

def create_experiment_dir(path):
    # path/train/config.json, train_loss, validation_loss.json
    os.makedirs(path, exist_ok=True)
    os.makedirs(path + '/train', exist_ok=True)
    os.makedirs(path + '/eval', exist_ok=True)
    
    # train_path = os.path.join(path,'train')
    eval_path = os.path.join(path,'eval')
    os.system('touch {}/classification_report.txt'.format(eval_path))
    # os.system('touch {}/validation_loss.json'.format(train_path))
    # os.system('touch {}/config.json'.format(train_path))
    # os.system('touch {}/feature_label.json'.format(eval_path))

def read_off(filename):
    points = []
    faces = []
    with open(filename, 'r') as f:
        first = f.readline()
        if (len(first) > 4): 
            n, m, c = first[3:].split(' ')[:]
        else:
            n, m, c = f.readline().rstrip().split(' ')[:]
        n = int(n)
        m = int(m)
        for i in range(n):
            value = f.readline().rstrip().split(' ')
            points.append([float(x) for x in value])
        for i in range(m):
            value = f.readline().rstrip().split(' ')
            faces.append([int(x) for x in value])
    points = np.array(points)
    faces = np.array(faces)
    return points, faces

def vis_point_cloud(points):
    pcl = o3d.geometry.PointCloud()
    pcl.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([pcl])

#TODO read all points!
def read_cloud_from_txt(cloud_path):
    points_list = [] 
    with open(cloud_path, 'r') as f:
        while True:
            lines = f.readline()
            
            if not lines:
                break
            # point = [float(i) for i in lines.split(',')[0:3]]
            point = [float(i) for i in lines.split(',')]
            points_list.append(point)
    return np.asarray(points_list, dtype = np.float32).T
