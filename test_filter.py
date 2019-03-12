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
from model_siamese import ft_net_dense_filter, ft_net_dense, SiameseNet, load_network_easy
from torchvision.datasets.folder import default_loader
######################################################################
# Options
# --------

parser = argparse.ArgumentParser(description='Testing')
parser.add_argument('--which_epoch', default='best_filter751', type=str, help='0,1,2,3...or last')
parser.add_argument('--test_dir', default='data/market/pytorch', type=str, help='./test_data')
parser.add_argument('--name', default='filter_model', type=str, help='save model path')
parser.add_argument('--batchsize', default=1024, type=int, help='batchsize')

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
        img_up = torch.cat((img[:, :int(img.size(1)/2)], img[:, :int(img.size(1)/2)]), 1)
        img_down = torch.cat((img[:, int(img.size(1)/2):], img[:, int(img.size(1)/2):]), 1)
        return img_up, img_down, int(label), file_name
        # return img, img, int(label), file_name


data_dir = test_dir
dataset_list = ['train_all_751']
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
    cnt_1 = 0
    cnt_2 = 0
    for phase in dataset_list:
        for data in dataloaders[phase]:
            inputs1, inputs2, id_labels, file_name = data
            if use_gpu:
                inputs1 = inputs1.cuda()
                inputs2 = inputs2.cuda()
            # forward
            with torch.no_grad():
                output1, output2, \
                result, result, result, result, result, result, result, result\
                    = model(inputs1, inputs2)
                _, id_preds1 = torch.max(output1.detach(), 1)
                _, id_preds2 = torch.max(output2.detach(), 1)
                id_labels = id_labels.cuda()
            loss1 = criterion(output1, id_labels)
            loss2 = criterion(output2, id_labels)
            # statistics
            running_loss += loss1.item()  # * opt.batchsize
            running_loss += loss2.item()  # * opt.batchsize
            running_corrects += float(torch.sum(id_preds1 == id_labels.detach()))
            running_corrects += float(torch.sum(id_preds2 == id_labels.detach()))
            ratio = 0.005
            batch_bad_num = int(ratio * inputs1.size(0))
            # largest=True mean select similar to real, otherwise fake
            p1 = [F.softmax(output1, 1)[i, id_labels[i]] for i in range(output1.size(0))]
            p1, index1 = torch.sort(torch.Tensor(p1), descending=False)
            p2 = [F.softmax(output2, 1)[i, id_labels[i]] for i in range(output2.size(0))]
            p2, index2 = torch.sort(torch.Tensor(p2), descending=False)
            output = output1 + output2
            p = [F.softmax(output, 1)[i, id_labels[i]] for i in range(output.size(0))]
            p, index = torch.sort(torch.Tensor(p), descending=False)

            for i in range(len(index1)):
                if i < batch_bad_num:
                    # shutil.copy(file_name[index1[i]], os.path.join(sample_bad, os.path.split(file_name[index1[i]])[-1]))
                    # shutil.copy(file_name[index2[i]], os.path.join(sample_bad, os.path.split(file_name[index2[i]])[-1]))
                    shutil.copy(file_name[index[i]], os.path.join(sample_bad, os.path.split(file_name[index[i]])[-1]))
                    cnt_1 += 1
                # elif i >= len(index1) - batch_bad_num:
                else:
                    # shutil.copy(file_name[index1[i]], os.path.join(sample_good, os.path.split(file_name[index1[i]])[-1]))
                    # shutil.copy(file_name[index2[i]], os.path.join(sample_good, os.path.split(file_name[index2[i]])[-1]))
                    shutil.copy(file_name[index[i]], os.path.join(sample_good, os.path.split(file_name[index[i]])[-1]))
                    cnt_2 += 1

            # index1 = (id_preds1 == id_labels.detach())
            # index2 = (id_preds2 == id_labels.detach())
            # for i in range(len(index1)):
            #     if index1[i].detach() == 0 and index2[i].detach() == 0:
            #         cnt_1 += 1
            #         shutil.copy(file_name[i], os.path.join(sample_bad, os.path.split(file_name[i])[-1]))
            #     else:
            #         shutil.copy(file_name[i], os.path.join(sample_good, os.path.split(file_name[i])[-1]))
            #     if index1[i].detach() == 0 or index2[i].detach() == 0:
            #         cnt_2 += 1
            print(cnt_1, cnt_2)


        datasize = dataset_sizes[phase]
        print('datasize = %d' % datasize)
        print('good_size = %d  bad_size = %d' % (len(os.listdir(sample_good)), len(os.listdir(sample_bad))))
        epoch_loss = running_loss / datasize
        epoch_acc = running_corrects / (datasize * 2)

        print('{} Loss: {:.4f}  Acc: {:.4f} '.format(phase, epoch_loss, epoch_acc))
        print('cnt1 = %d   cnt2 = %d' % (cnt_1, cnt_2))


def pack_to_dir():
    files = os.listdir(sample_good)
    print('original file num = %d' % len(files))
    dir = 'train_all_filter'
    dst_base_path = os.path.join(os.path.split(sample_good)[0], dir)
    if os.path.exists(dst_base_path):
        shutil.rmtree(dst_base_path)
    os.makedirs(dst_base_path)
    dir_num = 0
    file_num = 0
    for file in files:
        sub_dir = file[:4]
        if not os.path.exists(os.path.join(dst_base_path, sub_dir)):
            os.makedirs(os.path.join(dst_base_path, sub_dir))
            dir_num += 1
        shutil.copy(os.path.join(sample_good, file), os.path.join(dst_base_path, sub_dir, file))
        file_num += 1
    print('dir_num = %d   file_num = %d' % (dir_num, file_num))

######################################################################
# Load Collected data Trained model
print('-------test-----------')
class_num = len(os.listdir(os.path.join(opt.test_dir, 'train_all')))
embedding_net = ft_net_dense(751)
model_siamese = SiameseNet(embedding_net)
model_siamese = load_network_easy(model_siamese, name, opt.which_epoch)
model_siamese = model_siamese.eval()
if use_gpu:
    model = model_siamese.cuda()

criterion = nn.CrossEntropyLoss()
test(model_siamese, criterion)
pack_to_dir()