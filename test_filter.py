# -*- coding: utf-8 -*-

from __future__ import print_function, division
import argparse
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import os
import numpy as np
import torch.nn.functional as F
import shutil
import yaml
from model_siamese import ft_net_dense_filter
from torchvision.datasets.folder import default_loader
######################################################################
# Options
# --------

parser = argparse.ArgumentParser(description='Testing')
parser.add_argument('--which_epoch', default='best_filter', type=str, help='0,1,2,3...or last')
parser.add_argument('--test_dir', default='data/filter_data', type=str, help='./test_data')
parser.add_argument('--name', default='filter_model', type=str, help='save model path')
parser.add_argument('--batchsize', default=256, type=int, help='batchsize')

opt = parser.parse_args()
opt.use_dense = True
print('opt = %s' % opt)
print('opt.which_epoch = %s' % opt.which_epoch)
print('opt.test_dir = %s' % opt.test_dir)
print('opt.name = %s' % opt.name)
print('opt.batchsize = %s' % opt.batchsize)
###load config###
name = opt.name
test_dir = opt.test_dir

######################################################################
# Load Data
# ---------
data_transforms = transforms.Compose([
    transforms.Resize((256, 128), interpolation=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class filter_dataset(datasets.ImageFolder):
    def __init__(self, root, transform):
        super(filter_dataset, self).__init__(root, transform)
        self.labels = np.array(self.imgs)[:, 1]
        self.data = np.array(self.imgs)[:, 0]

    def __getitem__(self, index):
        file_name = self.data[index]
        img, label = self.data[index], self.labels[index].item()
        img = default_loader(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, int(label), file_name


data_dir = test_dir
dataset_list = ['test_set']
image_datasets = {x: filter_dataset(os.path.join(data_dir, x), data_transforms) for x in dataset_list}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                              shuffle=False, num_workers=16) for x in dataset_list}
dataset_sizes = {x: len(image_datasets[x]) for x in dataset_list}
use_gpu = torch.cuda.is_available()

sample_good = 'data/filter_data/good'
sample_bad = 'data/filter_data/bad'
if os.path.exists(sample_good):
    shutil.rmtree(sample_good)
if os.path.exists(sample_bad):
    shutil.rmtree(sample_bad)
os.makedirs(sample_good)
os.makedirs(sample_bad)
def test(model, criterion):
    model.eval()
    running_loss = 0
    running_corrects = 0
    for phase in dataset_list:
        for data in dataloaders[phase]:
            inputs, id_labels, file_name = data
            now_batch_size, c, h, w = inputs.shape
            if now_batch_size < opt.batchsize:  # next epoch
                continue
            if use_gpu:
                inputs = inputs.cuda()
            # forward
            with torch.no_grad():
                output = model(inputs)[0]
                id_labels = id_labels.cuda()
            _, id_preds = torch.max(output.detach(), 1)
            loss = criterion(output, id_labels)
            # statistics
            running_loss += loss.item()  # * opt.batchsize
            running_corrects += float(torch.sum(id_preds == id_labels.detach()))
            batch_filter_num = int(opt.batchsize/10)
            # largest=True mean select similar to real, otherwise fake
            v, index = F.softmax(output, 1)[:, 1].topk(batch_filter_num, largest=True)
            for i in range(index.size(0)):
                print('1 = %3d file_name = %-70s   p = %.5f' % (i, file_name[index[i]], v[i]))
                shutil.copy(file_name[index[i]], os.path.join(sample_good, os.path.split(file_name[index[i]])[-1]))
            v, index = F.softmax(output, 1)[:, 1].topk(batch_filter_num, largest=False)
            for i in range(index.size(0)):
                print('1 = %3d file_name = %-70s   p = %.5f' % (i, file_name[index[i]], v[i]))
                shutil.copy(file_name[index[i]], os.path.join(sample_bad, os.path.split(file_name[index[i]])[-1]))
            exit()

        datasize = dataset_sizes[phase] // opt.batchsize * opt.batchsize
        epoch_loss = running_loss / datasize
        epoch_acc = running_corrects / datasize

        print('{} Loss: {:.4f}  Acc: {:.4f} '.format(phase, epoch_loss, epoch_acc))


######################################################################
def load_network_easy(network, name, model_name=None):
    if model_name == None:
        save_path = os.path.join('./model', name, 'net_%s.pth' % 'last_filter')
    else:
        save_path = os.path.join('./model', name, 'net_%s.pth' % model_name)
    print('load easy pretrained model: %s' % save_path)
    network.load_state_dict(torch.load(save_path))
    return network
# Load Collected data Trained model
print('-------test-----------')
model = ft_net_dense_filter(2)
model = load_network_easy(model, name, opt.which_epoch)
if use_gpu:
    model = model.cuda()

criterion = nn.CrossEntropyLoss()

model = test(model, criterion)

