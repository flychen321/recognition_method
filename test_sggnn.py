# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import scipy.io
import yaml
from model_siamese import ft_net, ft_net_dense, PCB, PCB_test
from model_siamese import Sggnn_siamese, Sggnn_gcn, SiameseNet
from model_siamese import load_network_easy, load_network, save_network, save_whole_network

# fp16
try:
    from apex.fp16_utils import *
except ImportError:  # will be 3.x series
    print(
        'This is not an error. If you want to use low precision, i.e., fp16, please install the apex with cuda support (https://github.com/NVIDIA/apex) and update pytorch to 1.0')
######################################################################
# Options
# --------

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids', default='0', type=str, help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--which_epoch', default='best_siamese', type=str, help='0,1,2,3...or last')
parser.add_argument('--test_dir', default='market', type=str, help='./test_data')
parser.add_argument('--name', default='sggnn', type=str, help='save model path')
parser.add_argument('--batchsize', default=48, type=int, help='batchsize')
parser.add_argument('--use_dense', action='store_true', help='use densenet121')
parser.add_argument('--PCB', action='store_true', help='use PCB')
parser.add_argument('--fp16', action='store_true', help='use fp16.')

opt = parser.parse_args()
opt.use_dense = True
print('opt = %s' % opt)
print('opt.gpu_ids = %s' % opt.gpu_ids)
print('opt.which_epoch = %s' % opt.which_epoch)
print('opt.test_dir = %s' % opt.test_dir)
print('opt.name = %s' % opt.name)
print('opt.batchsize = %s' % opt.batchsize)
print('opt.use_dense = %s' % opt.use_dense)
print('opt.PCB = %s' % opt.PCB)
print('opt.fp16 = %s' % opt.fp16)
###load config###
# load the training config
config_path = os.path.join('./model', 'sggnn', 'opts.yaml')
with open(config_path, 'r') as stream:
    config = yaml.load(stream)
opt.fp16 = False  # config['fp16']
opt.PCB = config['PCB']
opt.use_dense = config['use_dense']

str_ids = opt.gpu_ids.split(',')
# which_epoch = opt.which_epoch
# name = opt.name
name = 'sggnn'
data_dir = os.path.join('data', opt.test_dir, 'pytorch')
print('data_dir = %s' % data_dir)

gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >= 0:
        gpu_ids.append(id)

# set gpu ids
# if len(gpu_ids) > 0:
#     torch.cuda.set_device(gpu_ids[0])
#     cudnn.benchmark = True

######################################################################
# Load Data
# ---------
#
# We will use torchvision and torch.utils.data packages for loading the
# data.
#
data_transforms = transforms.Compose([
    transforms.Resize((256, 128), interpolation=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

if opt.PCB:
    data_transforms = transforms.Compose([
        transforms.Resize((384, 192), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

# dataset_list = ['gallery', 'query', 'train_all']
dataset_list = ['gallery', 'query']
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms) for x in dataset_list}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                              shuffle=False, num_workers=16) for x in dataset_list}
class_names = image_datasets[dataset_list[1]].classes
use_gpu = torch.cuda.is_available()


######################################################################
# Extract feature
# ----------------------
#
# Extract feature from  a trained model.
#
def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()  # N x C x H x W
    img_flip = img.index_select(3, inv_idx)
    return img_flip


def extract_feature(model, dataloaders):
    features = torch.FloatTensor()
    count = 0
    for data in dataloaders:
        img, label = data
        n, c, h, w = img.size()
        count += n
        # print(count)
        if opt.use_dense:
            ff = torch.FloatTensor(n, 512).zero_().cuda()
        else:
            ff = torch.FloatTensor(n, 512).zero_().cuda()
        if opt.PCB:
            ff = torch.FloatTensor(n, 2048, 6).zero_().cuda()  # we have six parts
        for i in range(2):
            if (i == 1):
                img = fliplr(img)
            input_img = img.cuda()
            if opt.fp16:
                input_img = input_img.half()
            _, outputs = model(input_img)
            ff = ff + outputs
        # norm feature
        if opt.PCB:
            # feature size (n,2048,6)
            # 1. To treat every part equally, I calculate the norm for every 2048-dim part feature.
            # 2. To keep the cosine score==1, sqrt(6) is added to norm the whole feature (2048*6).
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * np.sqrt(6)
            ff = ff.div(fnorm.expand_as(ff))
            ff = ff.view(ff.size(0), -1)
        else:
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))

        ff = ff.detach().cpu().float()
        features = torch.cat((features, ff), 0)
    return features


def get_id(img_path):
    camera_id = []
    labels = []
    for path, v in img_path:
        # filename = path.split('/')[-1]
        filename = os.path.basename(path)
        label = filename[0:4]
        camera = filename.split('c')[1]
        if label[0:2] == '-1':
            labels.append(-1)
        else:
            labels.append(int(label))
        camera_id.append(int(camera[0]))
    return camera_id, labels


dataset_path = []
for i in range(len(dataset_list)):
    dataset_path.append(image_datasets[dataset_list[i]].imgs)

dataset_cam = []
dataset_label = []
for i in range(len(dataset_list)):
    cam, label = get_id(dataset_path[i])
    dataset_cam.append(cam)
    dataset_label.append(label)

######################################################################
# Load Collected data Trained model
print('-------test-----------')
class_num = len(os.listdir(os.path.join(data_dir, 'train_all')))
embedding_net = ft_net_dense(class_num)
# embedding_net = ft_net_dense(2027)
# embedding_net = ft_net_dense(702)
model_siamese = SiameseNet(embedding_net)
model_siamese = load_network_easy(model_siamese, name, opt.which_epoch)
model_siamese = model_siamese.eval()
if use_gpu:
    model = model_siamese.cuda()

# Extract feature
dataset_feature = []
with torch.no_grad():
    for i in range(len(dataset_list)):
        dataset_feature.append(extract_feature(model, dataloaders[dataset_list[i]]))


if len(dataset_list) == 3:
    result = {'gallery_f': dataset_feature[0].numpy(), 'gallery_label': dataset_label[0], 'gallery_cam': dataset_cam[0],
              'query_f': dataset_feature[1].numpy(), 'query_label': dataset_label[1], 'query_cam': dataset_cam[1],
              'train_f': dataset_feature[2].numpy(), 'train_label': dataset_label[2], 'train_cam': dataset_cam[2]}
else:
    result = {'gallery_f': dataset_feature[0].numpy(), 'gallery_label': dataset_label[0], 'gallery_cam': dataset_cam[0],
              'query_f': dataset_feature[1].numpy(), 'query_label': dataset_label[1], 'query_cam': dataset_cam[1]}
scipy.io.savemat('pytorch_result.mat', result)
