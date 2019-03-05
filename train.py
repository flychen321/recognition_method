# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import torch.backends.cudnn as cudnn
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
import copy
from PIL import Image
import time
import os
# from reid_sampler import StratifiedSampler
from model_siamese import ft_net, ft_net_dense, PCB, verif_net
from model_siamese import Sggnn_siamese, Sggnn_gcn, SiameseNet, Sggnn_all
from random_erasing import RandomErasing
from datasets import TripletFolder, SiameseDataset, GcnDataset
import yaml
from shutil import copyfile
from losses import ContrastiveLoss, SigmoidLoss
from model_siamese import load_network_easy, load_network, save_network, save_whole_network
from model_gcn import Sggnn_prepare, GCN

version = torch.__version__

######################################################################
# Options
# --------
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids', default='0', type=str, help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--name', default='ft_ResNet50', type=str, help='output model name')
parser.add_argument('--save_model_name', default='', type=str, help='save_model_name')
parser.add_argument('--data_dir', default='data/market/pytorch', type=str, help='training dir path')
parser.add_argument('--train_all', action='store_true', help='use all training data')
parser.add_argument('--color_jitter', action='store_true', help='use color jitter in training')
parser.add_argument('--batchsize', default=24, type=int, help='batchsize')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--alpha', default=1.0, type=float, help='alpha')
parser.add_argument('--erasing_p', default=0.5, type=float, help='Random Erasing probability, in [0,1]')
parser.add_argument('--use_dense', action='store_true', help='use densenet121')
parser.add_argument('--PCB', action='store_true', help='use PCB+ResNet50')
parser.add_argument('--net_loss_model', default=0, type=int, help='net_loss_model')

opt = parser.parse_args()
opt.use_dense = True
print('opt = %s' % opt)
print('net_loss_model = %d' % opt.net_loss_model)
print('save_model_name = %s' % opt.save_model_name)
data_dir = opt.data_dir
# name = opt.name
name = 'sggnn'
str_ids = opt.gpu_ids.split(',')
gpu_ids = []
for str_id in str_ids:
    gid = int(str_id)
    if gid >= 0:
        gpu_ids.append(gid)

# set gpu ids
if len(gpu_ids) > 0:
    torch.cuda.set_device(gpu_ids[0])
    cudnn.benchmark = True
# print(gpu_ids[0])


######################################################################
# Load Data
# ---------
#

transform_train_list = [
    # transforms.RandomResizedCrop(size=128, scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3), #Image.BICUBIC)
    transforms.Resize((256, 128), interpolation=3),
    transforms.Pad(10),
    transforms.RandomCrop((256, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]

transform_val_list = [
    transforms.Resize(size=(256, 128), interpolation=3),  # Image.BICUBIC
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]

if opt.PCB:
    transform_train_list = [
        transforms.Resize((384, 192), interpolation=3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
    transform_val_list = [
        transforms.Resize(size=(384, 192), interpolation=3),  # Image.BICUBIC
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]

if opt.erasing_p > 0:
    transform_train_list = transform_train_list + [RandomErasing(probability=opt.erasing_p, mean=[0.0, 0.0, 0.0])]

if opt.color_jitter:
    transform_train_list = [transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1,
                                                   hue=0)] + transform_train_list

print(transform_train_list)
data_transforms = {
    'train': transforms.Compose(transform_train_list),
    'val': transforms.Compose(transform_val_list),
}

train_all = ''
if opt.train_all:
    train_all = '_all'

image_datasets = {}

# dataset = TripletFolder
dataset = SiameseDataset
image_datasets['train'] = dataset(os.path.join(data_dir, 'train_all'),
                                  data_transforms['train'])
image_datasets['val'] = dataset(os.path.join(data_dir, 'val'),
                                data_transforms['val'])

dataloaders_sgg = {}
dataloaders_sgg['train'] = torch.utils.data.DataLoader(
    GcnDataset(os.path.join(data_dir, 'train_all'), data_transforms['train'], img_num=4), batch_size=opt.batchsize, shuffle=True,
    num_workers=8)

dataloaders_gcn = {}
dataloaders_gcn['train'] = torch.utils.data.DataLoader(
    GcnDataset(os.path.join(data_dir, 'train_all'), data_transforms['train'], img_num=2), batch_size=opt.batchsize, shuffle=True,
    num_workers=8)

batch = {}

class_names = image_datasets['train'].classes
class_vector = [s[1] for s in image_datasets['train'].samples]
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                              shuffle=True, num_workers=8)
               for x in ['train', 'val']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

use_gpu = torch.cuda.is_available()

since = time.time()
# inputs, classes, pos, pos_classes = next(iter(dataloaders['train']))
print(time.time() - since)

######################################################################
# Training the model
# ------------------
#
# Now, let's write a general function to train a model. Here, we will
# illustrate:
#
# -  Scheduling the learning rate
# -  Saving the best model
#
# In the following, parameter ``scheduler`` is an LR scheduler object from
# ``torch.optim.lr_scheduler``.

y_loss = {}  # loss history
y_loss['train'] = []
y_loss['val'] = []
y_err = {}
y_err['train'] = []
y_err['val'] = []


def train_model_triplet(model, model_verif, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0
    best_loss = 10000.0
    best_epoch = -1
    last_margin = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_verif_loss = 0.0
            running_corrects = 0.0
            running_verif_corrects = 0.0
            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                inputs, labels, pos, neg = data
                now_batch_size, c, h, w = inputs.shape

                if now_batch_size < opt.batchsize:  # next epoch
                    continue

                if use_gpu:
                    inputs = inputs.cuda()
                    pos = pos.cuda()
                    neg = neg.cuda()
                    labels = labels.cuda()

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs, f = model(inputs)
                _, pf = model(pos)
                _, nf = model(neg)
                # pscore = model_verif(pf * f)
                # nscore = model_verif(nf * f)
                pscore = model_verif((pf - f).pow(2))
                nscore = model_verif((nf - f).pow(2))
                # print(pf.requires_grad)
                # loss
                # ---------------------------------
                labels_0 = torch.zeros(now_batch_size).long()
                labels_1 = torch.ones(now_batch_size).long()
                labels_0 = labels_0.cuda()
                labels_1 = labels_1.cuda()

                _, preds = torch.max(outputs.detach(), 1)
                _, p_preds = torch.max(pscore.detach(), 1)
                _, n_preds = torch.max(nscore.detach(), 1)
                loss_id = criterion(outputs, labels)
                loss_verif = (criterion(pscore, labels_0) + criterion(nscore, labels_1)) * 0.5 * opt.alpha
                if opt.net_loss_model == 0:
                    loss = loss_id + loss_verif
                elif opt.net_loss_model == 1:
                    loss = loss_verif
                elif opt.net_loss_model == 2:
                    loss = loss_id
                else:
                    print('opt.net_loss_model = %s    error !!!' % opt.net_loss_model)
                    exit()
                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                # statistics
                if int(version[0]) > 0 or int(version[2]) > 3:  # for the new version like 0.4.0 and 0.5.0
                    running_loss += loss.item()  # * opt.batchsize
                    running_verif_loss += loss_verif.item()  # * opt.batchsize
                else:  # for the old version like 0.3.0 and 0.3.1
                    running_loss += loss.item()
                    running_verif_loss += loss_verif.item()
                running_corrects += float(torch.sum(preds == labels.detach()))
                running_verif_corrects += float(torch.sum(p_preds == 0)) + float(torch.sum(n_preds == 1))

            datasize = dataset_sizes['train'] // opt.batchsize * opt.batchsize
            epoch_loss = running_loss / datasize
            epoch_verif_loss = running_verif_loss / datasize
            epoch_acc = running_corrects / datasize
            epoch_verif_acc = running_verif_corrects / (2 * datasize)

            print('{} Loss: {:.4f} Loss_verif: {:.4f}  Acc: {:.4f} Verif_Acc: {:.4f} '.format(
                phase, epoch_loss, epoch_verif_loss, epoch_acc, epoch_verif_acc))
            # if phase == 'val':
            #     if epoch_acc > best_acc or (np.fabs(epoch_acc - best_acc) < 1e-5 and epoch_loss < best_loss):
            #         best_acc = epoch_acc
            #         best_loss = epoch_loss
            #         best_epoch = epoch
            #         best_model_wts = model.state_dict()
            #     if epoch >= 0:
            #         save_network(model, name, epoch)

            y_loss[phase].append(epoch_loss)
            y_err[phase].append(1.0 - epoch_acc)
            # deep copy the model
            epoch_acc = (epoch_acc + epoch_verif_acc) / 2.0
            epoch_loss = (epoch_loss + epoch_verif_loss) / 2.0
            if epoch_acc > best_acc or (np.fabs(epoch_acc - best_acc) < 1e-5 and epoch_loss < best_loss):
                best_acc = epoch_acc
                best_loss = epoch_loss
                best_epoch = epoch
                save_network(model, name, 'best')

            if epoch % 10 == 9:
                save_network(model, name, epoch)
            draw_curve(epoch)
            last_model_wts = model.state_dict()

        print()

    time_elapsed = time.time() - since
    print('best_epoch = %s     best_loss = %s     best_acc = %s' % (best_epoch, best_loss, best_acc))
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    # print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(last_model_wts)
    save_network(model, name, 'last')
    return model


def train_model_siamese(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0
    best_loss = 10000.0
    best_epoch = -1
    mse_criterion = nn.MSELoss()
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_id_loss = 0.0
            running_verif_loss = 0.0
            running_space_loss = 0.0
            running_id_corrects = 0.0
            running_verif_corrects = 0.0
            running_verif_corrects1 = 0.0
            running_verif_corrects2 = 0.0
            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                inputs, vf_labels, id_labels = data
                now_batch_size, c, h, w = inputs[0].shape
                if now_batch_size < opt.batchsize:  # next epoch
                    continue

                if type(inputs) not in (tuple, list):
                    inputs = (inputs,)
                if type(id_labels) not in (tuple, list):
                    id_labels = (id_labels,)
                if use_gpu:
                    inputs = tuple(d.cuda() for d in inputs)
                    id_labels = tuple(d.cuda() for d in id_labels)
                    if vf_labels is not None:
                        vf_labels = vf_labels.cuda()

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # outputs1, f1, outputs2, f2, feature, score = model(inputs[0], inputs[1])
                # _, id_preds1 = torch.max(outputs1.detach(), 1)
                # _, id_preds2 = torch.max(outputs2.detach(), 1)
                # _, vf_preds = torch.max(score.detach(), 1)
                # loss_id1 = criterion(outputs1, id_labels[0])
                # loss_id2 = criterion(outputs2, id_labels[1])
                # loss_id = loss_id1 + loss_id2
                # loss_verif = criterion(score, vf_labels)

                # opt.net_loss_model = 0
                # if opt.net_loss_model == 0:
                #     loss = loss_id + loss_verif
                # elif opt.net_loss_model == 1:
                #     loss = loss_verif
                # elif opt.net_loss_model == 2:
                #     loss = loss_id
                # else:
                #     print('opt.net_loss_model = %s    error !!!' % opt.net_loss_model)
                #     exit()

                # outputs1, outputs2, result, result_combine_1, result_combine_2 = model(inputs[0], inputs[1])
                output1, output2, \
                result, result1_12, result1_21, result2_12, result2_21, result12_21,\
                feature_sum_orig, feature_sum_new = model(inputs[0], inputs[1])
                _, id_preds1 = torch.max(output1.detach(), 1)
                _, id_preds2 = torch.max(output2.detach(), 1)
                _, vf_preds = torch.max(result.detach(), 1)
                _, vf_preds1_12 = torch.max(result1_12.detach(), 1)
                _, vf_preds1_21 = torch.max(result1_21.detach(), 1)
                _, vf_preds2_12 = torch.max(result2_12.detach(), 1)
                _, vf_preds2_21 = torch.max(result2_21.detach(), 1)
                _, vf_preds12_21 = torch.max(result12_21.detach(), 1)
                loss_id1 = criterion(output1, id_labels[0])
                loss_id2 = criterion(output2, id_labels[1])
                loss_id = loss_id1 + loss_id2
                loss_verif0 = criterion(result, vf_labels)
                loss_verif1 = criterion(result1_12, vf_labels)
                loss_verif2 = criterion(result1_21, vf_labels)
                loss_verif3 = criterion(result2_12, vf_labels)
                loss_verif4 = criterion(result2_21, vf_labels)
                loss_verif5 = criterion(result12_21, vf_labels)
                loss_verif = loss_verif0 + (loss_verif1 + loss_verif2 + loss_verif3 + loss_verif4 + loss_verif5)/5.0
                loss_space = mse_criterion(feature_sum_orig, feature_sum_new)
                if opt.net_loss_model == 0:
                    r1 = 0.66
                    r2 = 0.33
                    r3 = 0.00
                elif opt.net_loss_model == 1:
                    r1 = 0.5
                    r2 = 0.5
                    r3 = 0.01
                elif opt.net_loss_model == 2:
                    r1 = 0.66
                    r2 = 0.33
                    r3 = 0.01
                loss = r1 * loss_id + r2 * loss_verif + r3 * loss_space

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                # statistics
                running_id_loss += loss_id.item()  # * opt.batchsize
                running_verif_loss += loss_verif.item()  # * opt.batchsize
                running_space_loss += loss_space.item()

                running_id_corrects += float(torch.sum(id_preds1 == id_labels[0].detach()))
                running_id_corrects += float(torch.sum(id_preds2 == id_labels[1].detach()))
                running_verif_corrects += float(torch.sum(vf_preds == vf_labels))
                running_verif_corrects += float(torch.sum(vf_preds1_12 == vf_labels))
                running_verif_corrects += float(torch.sum(vf_preds1_21 == vf_labels))
                running_verif_corrects += float(torch.sum(vf_preds2_12 == vf_labels))
                running_verif_corrects += float(torch.sum(vf_preds2_21 == vf_labels))
                running_verif_corrects += float(torch.sum(vf_preds12_21 == vf_labels))

            datasize = dataset_sizes['train'] // opt.batchsize * opt.batchsize
            epoch_id_loss = running_id_loss / datasize
            epoch_verif_loss = running_verif_loss / datasize
            epoch_space_loss = running_space_loss / datasize
            epoch_id_acc = running_id_corrects / (datasize * 2)
            epoch_verif_acc = running_verif_corrects / (datasize * 6)

            print('{} Loss_id: {:.4f} Loss_verify: {:.4f} Loss_space: {:.4f}  Acc_id: {:.4f} Acc_verify: {:.4f} '.format(
                phase, epoch_id_loss, epoch_verif_loss, epoch_space_loss, epoch_id_acc, epoch_verif_acc))

            epoch_acc = (epoch_id_acc + epoch_verif_acc) / 2.0
            epoch_loss = (epoch_id_loss + epoch_verif_loss) / 2.0
            if epoch_acc > best_acc or (np.fabs(epoch_acc - best_acc) < 1e-5 and epoch_loss < best_loss):
                best_acc = epoch_acc
                best_loss = epoch_loss
                best_epoch = epoch
                # best_model_wts = model.state_dict()
                save_network(model, name, 'best_siamese')
                save_network(model, name, 'best_siamese_' + str(opt.save_model_name))
                save_whole_network(model, name, 'whole_best_siamese')

            y_loss[phase].append(epoch_id_loss)
            y_err[phase].append(1.0 - epoch_id_acc)
            # deep copy the model

            if epoch % 10 == 9:
                save_network(model, name, epoch)

            draw_curve(epoch)
            # last_model_wts = model.state_dict()

    time_elapsed = time.time() - since
    print('best_epoch = %s     best_loss = %s     best_acc = %s' % (best_epoch, best_loss, best_acc))
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # load best model weights
    # model.load_state_dict(best_model_wts)
    # save_network(model, name, 'best_siamese')
    # save_network(model, name, 'best_siamese_' + str(opt.save_model_name))
    # save_whole_network(model, name, 'whole_best_siamese')
    # load last model weights
    # model.load_state_dict(last_model_wts)
    save_network(model, name, 'last_siamese')
    save_network(model, name, 'last_siamese_' + str(opt.save_model_name))
    save_whole_network(model, name, 'whole_last_siamese')
    return model


def train_gcn(train_loader, model_siamese, model_gcn, loss_gcn_fn, optimizer_gcn, scheduler_gcn, num_epochs=25):
    global cnt
    since = time.time()
    model_gcn.train(True)
    model_siamese.eval()
    losses = []
    total_loss = 0
    for epoch in range(num_epochs):
        scheduler_gcn.step()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        running_corrects = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            target = target if len(target) > 0 else None
            if not type(data) in (tuple, list):
                data = (data,)
            if use_gpu:
                data = tuple(d.cuda() for d in data)
                if target is not None:
                    target = target.cuda()

            with torch.no_grad():
                adj, feature, target1 = model_siamese(*data, target)
            # for batch_idx in range(100000):
            optimizer_gcn.zero_grad()
            target = target1
            outputs = model_gcn(feature, adj)  # for SGGNN_GCN
            outputs_org = outputs
            _, preds = torch.max(outputs.detach(), 1)
            running_corrects = float(torch.sum(preds == target.detach()))
            epoch_id_acc = running_corrects / len(preds)

            if type(outputs) not in (tuple, list):
                outputs = (outputs,)

            loss_inputs = outputs
            if target is not None:
                target = (target,)
                loss_inputs += target

            loss_inputs = tuple(d.cuda() for d in loss_inputs)

            loss_outputs = loss_gcn_fn(*loss_inputs)
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            losses.append(loss.item())
            total_loss += loss.item()
            loss.backward()
            optimizer_gcn.step()
            if batch_idx % 50 == 0:
                print('epoch = %2d  batch_idx = %4d  loss = %.5f' % (epoch, batch_idx, loss))
                print('acc = %.5f' % (epoch_id_acc))
                print('preds  = %s' % preds[:20])
                print('target = %s' % target[0][:20])
                print('outputs_org = %s' % outputs_org[:10, :])
        save_network(model_gcn, name, 'gcn' + str(epoch))
        save_whole_network(model_gcn, name, 'whole_gcn' + str(epoch))
    time_elapsed = time.time() - since
    print('time = %f' % (time_elapsed))
    save_network(model_gcn, name, 'last_gcn')
    save_whole_network(model_gcn, name, 'whole_last_gcn')
    return model_gcn


x_epoch = []
fig = plt.figure()
ax0 = fig.add_subplot(121, title="triplet_loss")
ax1 = fig.add_subplot(122, title="top1_err")


def draw_curve(current_epoch):
    x_epoch.append(current_epoch)
    ax0.plot(x_epoch, y_loss['train'], 'bo-', label='train')
    #    ax0.plot(x_epoch, y_loss['val'], 'ro-', label='val')
    ax1.plot(x_epoch, y_err['train'], 'bo-', label='train')
    #    ax1.plot(x_epoch, y_err['val'], 'ro-', label='val')
    if current_epoch == 0:
        ax0.legend()
        ax1.legend()
    fig.savefig(os.path.join('./model', name, 'train.jpg'))


######################################################################
# Train and evaluate
# ^^^^^^^^^^^^^^^^^^
#
# It should take around 1-2 hours on GPU. 
#
dir_name = os.path.join('./model', name)
if not os.path.exists('model'):
    os.mkdir('model')
if not os.path.isdir(dir_name):
    os.mkdir(dir_name)
    # copyfile('./train.py', dir_name + '/train.py')
    # copyfile('./model_siamese.py', dir_name + '/model_siamese.py')
    # copyfile('./datasets.py', dir_name + '/datasets.py')

# save opts
with open('%s/opts.yaml' % dir_name, 'w') as fp:
    yaml.dump(vars(opt), fp, default_flow_style=False)

stage_1 = True
stage_2 = False
stage_3 = False

if stage_1:
    print('class_num = %d' % len(class_names))
    embedding_net = ft_net_dense(len(class_names))
    model_siamese = SiameseNet(embedding_net)
    if use_gpu:
        model_siamese.cuda()
    print('model_siamese structure')
    # print(model_siamese)
    criterion = nn.CrossEntropyLoss()

    stage_1_classifier_id = list(map(id, model_siamese.embedding_net.classifier.parameters())) \
                            + list(map(id, model_siamese.embedding_net.model.fc.parameters()))
    stage_1_verify_id = list(map(id, model_siamese.classifier.parameters())) \
        # + list(map(id, model_siamese.bn.parameters()))
    stage_1_classifier_params = filter(lambda p: id(p) in stage_1_classifier_id, model_siamese.parameters())
    stage_1_verify_params = filter(lambda p: id(p) in stage_1_verify_id, model_siamese.parameters())
    stage_1_base_params = filter(lambda p: id(p) not in stage_1_classifier_id + stage_1_verify_id,
                                 model_siamese.parameters())

    optimizer_ft = optim.SGD([
        {'params': stage_1_base_params, 'lr': 0.1 * opt.lr},
        {'params': stage_1_classifier_params, 'lr': 1 * opt.lr},
        {'params': stage_1_verify_params, 'lr': 1 * opt.lr},
    ], weight_decay=5e-4, momentum=0.9, nesterov=True)

    # exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_ft, milestones=[40, 60], gamma=0.1)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=15, gamma=0.32)
    model = train_model_siamese(model_siamese, criterion, optimizer_ft, exp_lr_scheduler,
                                num_epochs=100)

if stage_2:
    margin = 1.
    embedding_net = ft_net_dense(len(class_names))
    model_mid = SiameseNet(embedding_net)
    print('model_mid_original = %s' % model_mid.embedding_net.model.features.conv0.weight[0, 0, 0])
    model_mid = load_network_easy(model_mid, name)
    print('model_mid_new = %s' % model_mid.embedding_net.model.features.conv0.weight[0, 0, 0])
    model_siamese = Sggnn_siamese(model_mid, hard_weight=False)
    print('model_mid_new2 = %s' % model_siamese.basemodel.embedding_net.model.features.conv0.weight[0, 0, 0])
    model_gcn = Sggnn_gcn()
    # model_gcn = load_network_easy(model_gcn, name, 'best_gcn')
    print('model_gcn = %s' % model_gcn.rf.fc[0].weight[0][:5])

    if use_gpu:
        model_siamese.cuda()
        model_gcn.cuda()
    # model_siamese = load_network(model_siamese, name)
    # loss_siamese_fn = ContrastiveLoss(margin)
    loss_siamese_fn = nn.CrossEntropyLoss()
    loss_gcn_fn = nn.CrossEntropyLoss()
    lr = 1e-3
    optimizer_siamese = optim.Adam(model_siamese.parameters(), lr=lr)
    scheduler_siamese = lr_scheduler.StepLR(optimizer_siamese, 8, gamma=0.1, last_epoch=-1)
    optimizer_gcn = optim.Adam(model_gcn.parameters(), lr=lr)
    scheduler_gcn = lr_scheduler.StepLR(optimizer_gcn, 2, gamma=0.1, last_epoch=-1)
    n_epochs = 5
    log_interval = 100
    model = train_gcn(dataloaders_sgg['train'], model_siamese, loss_siamese_fn, optimizer_siamese, scheduler_siamese,
                      model_gcn, loss_gcn_fn, optimizer_gcn, scheduler_gcn, num_epochs=n_epochs)

if stage_3:
    embedding_net = ft_net_dense(len(class_names))
    model_mid = SiameseNet(embedding_net)
    print('model_mid_original = %s' % model_mid.embedding_net.model.features.conv0.weight[0, 0, 0])
    model_mid = load_network_easy(model_mid, name)
    print('model_mid_new = %s' % model_mid.embedding_net.model.features.conv0.weight[0, 0, 0])
    model_siamese = Sggnn_prepare(model_mid, hard_weight=False)
    print('model_mid_new2 = %s' % model_siamese.basemodel.embedding_net.model.features.conv0.weight[0, 0, 0])

    model_gcn = GCN(nfeat=512,
                    nhid=64,
                    nclass=2,
                    dropout=0.5)
    optimizer_gcn = optim.Adam(model_gcn.parameters(),
                           lr=0.001, weight_decay=5e-4)

    if use_gpu:
        model_siamese.cuda()
        model_gcn.cuda()
    loss_gcn_fn = nn.CrossEntropyLoss()
    # loss_gcn_fn = nn.MSELoss()
    # loss_gcn_fn = F.nll_loss
    # lr = 1e-3
    # optimizer_gcn = optim.Adam(model_gcn.parameters(), lr=lr)
    scheduler_gcn = lr_scheduler.StepLR(optimizer_gcn, 20, gamma=0.3, last_epoch=-1)
    n_epochs = 100
    model = train_gcn(dataloaders_gcn['train'], model_siamese, model_gcn, loss_gcn_fn, optimizer_gcn, scheduler_gcn,
                      num_epochs=n_epochs)
