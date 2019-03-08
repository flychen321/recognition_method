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
import time
import os
from model_siamese import ft_net_dense_filter
from random_erasing import RandomErasing
import yaml
from model_siamese import save_network

version = torch.__version__

######################################################################
# Options
# --------
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--name', default='filter_model', type=str, help='output model name')
parser.add_argument('--save_model_name', default='', type=str, help='save_model_name')
parser.add_argument('--data_dir', default='data/filter_data', type=str, help='training dir path')
parser.add_argument('--train_all', action='store_true', help='use all training data')
parser.add_argument('--batchsize', default=96, type=int, help='batchsize')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--alpha', default=1.0, type=float, help='alpha')
parser.add_argument('--erasing_p', default=0.5, type=float, help='Random Erasing probability, in [0,1]')
parser.add_argument('--net_loss_model', default=0, type=int, help='net_loss_model')

opt = parser.parse_args()
print('opt = %s' % opt)
print('net_loss_model = %d' % opt.net_loss_model)
print('save_model_name = %s' % opt.save_model_name)
data_dir = opt.data_dir
name = opt.name

######################################################################
# Load Data
# ---------
#

transform_train_list = [
    transforms.Resize((256, 128), interpolation=3),
    # transforms.Pad(10),
    # transforms.RandomCrop((256, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]

transform_val_list = [
    transforms.Resize(size=(256, 128), interpolation=3),  # Image.BICUBIC
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]

if opt.erasing_p > 0:
    transform_train_list = transform_train_list + [RandomErasing(probability=opt.erasing_p, mean=[0.0, 0.0, 0.0])]

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
dataset = datasets.ImageFolder
image_datasets['train'] = dataset(os.path.join(data_dir, 'train_set'),
                                  data_transforms['train'])
image_datasets['val'] = dataset(os.path.join(data_dir, 'test_set'),
                                data_transforms['val'])


class_names = image_datasets['train'].classes
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                              shuffle=True, num_workers=8)
               for x in ['train', 'val']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

use_gpu = torch.cuda.is_available()

since = time.time()
print(time.time() - since)

######################################################################
# Training the model
# ------------------

y_loss = {}  # loss history
y_loss['train'] = []
y_loss['val'] = []
y_err = {}
y_err['train'] = []
y_err['val'] = []


def train(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    best_acc = 0.0
    best_loss = 10000.0
    best_epoch = -1
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
            running_corrects = 0.0
            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                inputs, id_labels = data
                now_batch_size, c, h, w = inputs.shape
                if now_batch_size < opt.batchsize:  # next epoch
                    continue

                if use_gpu:
                    inputs = inputs.cuda()
                    id_labels = id_labels.cuda()

                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                output = model(inputs)[0]
                _, id_preds = torch.max(output.detach(), 1)
                loss = criterion(output, id_labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                # statistics
                running_loss += loss.item()  # * opt.batchsize
                running_corrects += float(torch.sum(id_preds == id_labels.detach()))

            datasize = dataset_sizes['train'] // opt.batchsize * opt.batchsize
            epoch_loss = running_loss / datasize
            epoch_acc = running_corrects / datasize

            print('{} Loss: {:.4f}  Acc: {:.4f} '.format(phase, epoch_loss, epoch_acc))

            if epoch_acc > best_acc or (np.fabs(epoch_acc - best_acc) < 1e-5 and epoch_loss < best_loss):
                best_acc = epoch_acc
                best_loss = epoch_loss
                best_epoch = epoch
                save_network(model, name, 'best_filter')

            if epoch % 10 == 9:
                save_network(model, name, epoch)

    time_elapsed = time.time() - since
    print('best_epoch = %s     best_loss = %s     best_acc = %s' % (best_epoch, best_loss, best_acc))
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    save_network(model, name, 'last_filter')
    return model



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


print('class_num = %d' % len(class_names))
model = ft_net_dense_filter(len(class_names))
if use_gpu:
    model.cuda()
print('model structure')
print(model)

criterion = nn.CrossEntropyLoss()

classifier_id = list(map(id, model.classifier.parameters())) \
                        + list(map(id, model.model.fc.parameters()))
classifier_params = filter(lambda p: id(p) in classifier_id, model.parameters())
base_params = filter(lambda p: id(p) not in classifier_id, model.parameters())

optimizer_ft = optim.SGD([
    {'params': classifier_params, 'lr': 1 * opt.lr},
    {'params': base_params, 'lr': 0.1 * opt.lr},
], weight_decay=5e-4, momentum=0.9, nesterov=True)

# exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_ft, milestones=[40, 60], gamma=0.1)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=3, gamma=0.32)
model = train(model, criterion, optimizer_ft, exp_lr_scheduler,
                            num_epochs=10)


