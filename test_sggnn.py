# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
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
parser.add_argument('--which_epoch', default='best', type=str, help='0,1,2,3...or last')
parser.add_argument('--test_dir', default='data/market/pytorch', type=str, help='./test_data')
parser.add_argument('--name', default='sggnn', type=str, help='save model path')
parser.add_argument('--batchsize', default=512, type=int, help='batchsize')
parser.add_argument('--use_dense', action='store_true', help='use densenet121')
parser.add_argument('--PCB', action='store_true', help='use PCB')
parser.add_argument('--fp16', action='store_true', help='use fp16.')

opt = parser.parse_args()
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
config_path = os.path.join('./model', opt.name, 'opts.yaml')
with open(config_path, 'r') as stream:
    config = yaml.load(stream)
opt.fp16 = False  # config['fp16']
opt.PCB = config['PCB']
opt.use_dense = config['use_dense']

str_ids = opt.gpu_ids.split(',')
# which_epoch = opt.which_epoch
# name = opt.name
name = 'sggnn'
test_dir = opt.test_dir

gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >= 0:
        gpu_ids.append(id)

# set gpu ids
if len(gpu_ids) > 0:
    torch.cuda.set_device(gpu_ids[0])
    cudnn.benchmark = True

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

data_dir = test_dir

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms) for x in ['gallery', 'query']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                              shuffle=False, num_workers=16) for x in ['gallery', 'query']}
class_names = image_datasets['query'].classes
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
            input_img = Variable(img.cuda())
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

        ff = ff.data.cpu().float()
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


gallery_path = image_datasets['gallery'].imgs
query_path = image_datasets['query'].imgs

gallery_cam, gallery_label = get_id(gallery_path)
query_cam, query_label = get_id(query_path)


######################################################################
# Load Collected data Trained model
print('-------test-----------')
embedding_net = ft_net_dense(751)
model_siamese = SiameseNet(embedding_net)
model_siamese = load_network_easy(model_siamese, name)
model_siamese = model_siamese.eval()
if use_gpu:
    model = model_siamese.cuda()

# Extract feature
with torch.no_grad():
    gallery_feature = extract_feature(model, dataloaders['gallery'])
    query_feature = extract_feature(model, dataloaders['query'])


result = {'gallery_f': gallery_feature.numpy(), 'gallery_label': gallery_label, 'gallery_cam': gallery_cam,
          'query_f': query_feature.numpy(), 'query_label': query_label, 'query_cam': query_cam}
scipy.io.savemat('pytorch_result.mat', result)

