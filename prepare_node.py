import scipy.io
import torch
import numpy as np
import time
import os
import matplotlib
from torchvision import datasets, models, transforms

# dirs = os.listdir('data/market/pytorch/train_all')
# n = []
# for dir in dirs:
#     files = os.listdir(os.path.join('data/market/pytorch/train_all', dir))
#     n.append(len(files))
# n.sort()
# print(n)
# n = np.array(n)
# print(n.sum())
# n_s = n*n
# print(n_s.sum()/2)
######################################################################
np.random.seed(1)


def get_id(img_path):
    camera_id = []
    labels = []
    for path, v in img_path:
        filename = os.path.basename(path)
        label = filename[0:4]
        camera = filename.split('c')[1]
        if label[0:2] == '-1':
            labels.append(-1)
        else:
            labels.append(int(label))
        camera_id.append(int(camera[0]))
    return camera_id, labels


data_dir = 'data/market/pytorch/train_all'
image_datasets = datasets.ImageFolder(data_dir)
cams, labels = get_id(image_datasets.imgs)

result = scipy.io.loadmat('pytorch_result_market.mat')
query_feature = torch.FloatTensor(result['query_f'])
query_cam = result['query_cam'][0]
query_label = result['query_label'][0]
gallery_feature = torch.FloatTensor(result['gallery_f'])
gallery_cam = result['gallery_cam'][0]
gallery_label = result['gallery_label'][0]

train_feature = torch.FloatTensor(result['train_f'])
# train_feature = result['train_f']
train_cam = result['train_cam'][0]
train_label = result['train_label'][0]

# for i in range(len(train_label)):
#     if int(labels[i]) != train_label[i]:
#         print('labels[i] != train_label[i]: i = %4d  %d  %d' % (i, labels[i], train_label[i]))


num_total = len(train_label)
i = 0
n = []
feature_same = []
feature_dif = []

while i < num_total - 1:
    j = i
    while i < num_total - 1 and train_label[i] == train_label[i + 1]:
        i += 1
    i += 1
    k = i
    part_c = train_cam[j: k]
    part_l = train_label[j: k]
    part_index = np.arange(j, k)
    part_num = k - j
    part_index = np.random.permutation(part_index)
    if part_num % 2 != 0:
        part_index = part_index[:-1]
    former_index = part_index[:int(part_num / 2)]
    latter_index = part_index[int(part_num / 2):]
    for s in range(int(part_num / 2)):
        feature_same.append((train_feature[former_index[s]] - train_feature[latter_index[s]]).pow(2))
        if s != int(part_num / 2) - 1 - s:
            feature_same.append(
                (train_feature[former_index[s]] - train_feature[latter_index[int(part_num / 2) - 1 - s]]).pow(2))
    n.append(part_num)

i = 0
used_couple = []
while i < num_total - 1:
    j = i
    while i < num_total - 1 and train_label[i] == train_label[i + 1]:
        i += 1
    i += 1
    k = i
    part_c = train_cam[j: k]
    part_l = train_label[j: k]
    part_index = np.arange(j, k)
    part_num = k - j
    part_index = np.random.permutation(part_index)
    first_index = np.random.choice(part_index, int(len(part_index) / 2), replace=False)
    other_index = np.concatenate((np.arange(j), np.arange(k, num_total)), 0)
    second_index = np.random.choice(other_index, int(len(part_index) / 2 * 5), replace=False)
    for s in range(len(first_index)):
        for t in range(len(second_index)):
            if first_index[s] < second_index[t]:
                p = '%sa%s' % (first_index[s], second_index[t])
            else:
                p = '%sa%s' % (second_index[t], first_index[s])
            # if p not in used_couple:
            if True:
                feature_dif.append((train_feature[first_index[s]] - train_feature[second_index[t]]).pow(2))
                used_couple.append(p)

node_same = torch.Tensor(len(feature_same), len(feature_same[0]))
node_dif = torch.Tensor(len(feature_dif), len(feature_dif[0]))
print(node_same.shape)
print(node_dif.shape)
for i in range(len(feature_same)):
    node_same[i] = torch.Tensor(feature_same[i])
for i in range(len(feature_dif)):
    node_dif[i] = torch.Tensor(feature_dif[i])
dist_same = torch.sum(node_same, -1)
dist_dif = torch.sum(node_dif, -1)
dist_same_sorted = dist_same.sort()
node_same = node_same[dist_same_sorted[1]]
dist_dif_sorted = dist_dif.sort()
node_dif = node_dif[dist_dif_sorted[1]]
print(dist_same.shape)
print(dist_dif.shape)
print('len(feature_same) = %d' % (len(feature_same)))
print('len(feature_dif) = %d' % (len(feature_dif)))
result = {'feature_same': node_same.numpy(), 'feature_dif': node_dif.numpy(),
          'dist_same': dist_same.numpy(), 'dist_dif': dist_dif.numpy()}
scipy.io.savemat('nodes_info.mat', result)
exit()

n = np.array(n)
print(len(train_label))
print(n.sum())
n.sort()
print(n)

num = 2000
print('total num = %d   used_num = %d' % (num_total, num))

feature = train_feature[:num]
cam = train_cam[:num]
label = train_label[:num]

dist = torch.FloatTensor(num, num).zero_()
dist_f = torch.FloatTensor(num, num, 512).zero_()

use_gpu = torch.cuda.is_available()
if use_gpu:
    feature = feature.cuda()
    dist = dist.cuda()
    dist_f = dist_f.cuda()

for i in range(num):
    dist[i, :] = (feature[i] - feature).pow(2).sum(-1)
    dist_f[i, :] = (feature[i] - feature).pow(2)
