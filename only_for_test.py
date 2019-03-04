import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.io import savemat
import time
import random
import os
import math
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
#
# a = np.array([1,4,5,2,9,7])
# b = a.sort()
# print(a)
#
# baseline
# Rank@1:0.250000 Rank@5:0.490000 Rank@10:0.560000 mAP:0.128325
# random_walk
# Rank@1:0.400000 Rank@5:0.500000 Rank@10:0.630000 mAP:0.189465
# pos_neg_guider
# Rank@1:0.300000 Rank@5:0.470000 Rank@10:0.590000 mAP:0.131864
# pos
# Rank@1:0.260000 Rank@5:0.500000 Rank@10:0.570000 mAP:0.133315
# neg
# Rank@1:0.160000 Rank@5:0.310000 Rank@10:0.390000 mAP:0.081767
# gcn
# Rank@1:0.100000 Rank@5:0.250000 Rank@10:0.380000 mAP:0.054960
#
# norm baseline
# Rank@1:0.200000 Rank@5:0.360000 Rank@10:0.430000 mAP:0.114806
# Rank@1:0.070000 Rank@5:0.270000 Rank@10:0.410000 mAP:0.044516
# Rank@1:0.220000 Rank@5:0.360000 Rank@10:0.440000 mAP:0.109732
# Rank@1:0.310000 Rank@5:0.440000 Rank@10:0.500000 mAP:0.143855
a = []
input = torch.randn(48, 512)
for i in range(128):
    a.append(input[0][i*4: i*4+4].mean())
print(input.shape)
output = F.adaptive_avg_pool1d(input.unsqueeze(0), 128).squeeze(0)
# output = F.avg_pool1d(input.unsqueeze(0), 4)
# print(output.shape)
# output = output.squeeze(0)
print(output.shape)

import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
import os
import numpy as np
import math
import scipy.sparse as sp
import torch.nn.functional as F
######################################################################
# Load model
# ---------------------------
def load_network_easy(network, name, model_name=None):
    if model_name == None:
        save_path = os.path.join('./model', name, 'net_%s.pth' % 'last_siamese')
    else:
        save_path = os.path.join('./model', name, 'net_%s.pth' % model_name)
    print('load easy pretrained model: %s' % save_path)
    network.load_state_dict(torch.load(save_path))
    return network


def load_network(network, name, model_name=None):
    if model_name == None:
        save_path = os.path.join('./model', name, 'net_%s.pth' % 'whole_last_siamese')
    else:
        save_path = os.path.join('./model', name, 'net_%s.pth' % model_name)
    print('load whole pretrained model: %s' % save_path)
    net_original = torch.load(save_path)
    # print('pretrained = %s' % net_original.embedding_net.model.features.conv0.weight[0, 0, 0])
    pretrained_dict = net_original.state_dict()
    model_dict = network.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # print('network_original = %s' % network.embedding_net.model.features.conv0.weight[0, 0, 0])
    model_dict.update(pretrained_dict)
    network.load_state_dict(model_dict)
    # print('network_new = %s' % network.embedding_net.model.features.conv0.weight[0, 0, 0])

    # old = []
    # new = []
    # for k, v in pretrained_dict.items():
    #     old.append(k)
    # for k in model_dict:
    #     new.append(k)
    # print('len(old) = %d   len(new) = %d' % (len(old), len(new)))
    # for i in range(min(len(old), len(new))):
    #     print('i = %d  old = %s' % (i, old[i]))
    #     print('i = %d  new = %s' % (i, new[i]))
    # exit()
    return network


######################################################################
# Save model
# ---------------------------
def save_network(network, name, epoch_label):
    save_filename = 'net_%s.pth' % epoch_label
    save_path = os.path.join('./model', name, save_filename)
    torch.save(network.state_dict(), save_path)


def save_whole_network(network, name, epoch_label):
    save_filename = 'net_%s.pth' % epoch_label
    save_path = os.path.join('./model', name, save_filename)
    torch.save(network, save_path)


######################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.detach(), a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.detach(), a=0, mode='fan_out')
        init.constant_(m.bias.detach(), 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.detach(), 1.0, 0.02)
        init.constant_(m.bias.detach(), 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.detach(), std=0.001)
        init.constant_(m.bias.detach(), 0.0)


# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class Fc_ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, dropout=0.5, relu=False, num_bottleneck=512):
        super(Fc_ClassBlock, self).__init__()
        add_block = []
        add_block += [nn.Linear(input_dim, num_bottleneck)]
        add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        if dropout:
            classifier += [nn.Dropout(p=dropout)]
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        x = self.add_block(x)
        f = x
        f_norm = f.norm(p=2, dim=1, keepdim=True) + 1e-8
        f = f.div(f_norm)
        x = self.classifier(x)
        return x, f


class ReFineBlock(nn.Module):
    def __init__(self, input_dim=512, dropout=True, relu=True, num_bottleneck=512, layer=2):
        super(ReFineBlock, self).__init__()
        add_block = []
        for i in range(layer):
            add_block += [nn.Linear(input_dim, num_bottleneck)]
            add_block += [nn.BatchNorm1d(num_bottleneck)]
            if relu:
                add_block += [nn.LeakyReLU(0.1)]
            if dropout:
                add_block += [nn.Dropout(p=0.5)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)
        self.fc = add_block

    def forward(self, x):
        x = self.fc(x)
        return x


class FcBlock(nn.Module):
    def __init__(self, input_dim=512, dropout=True, relu=True, num_bottleneck=512):
        super(FcBlock, self).__init__()
        add_block = []
        add_block += [nn.Linear(input_dim, num_bottleneck)]
        add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if dropout:
            add_block += [nn.Dropout(p=0.5)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)
        self.fc = add_block

    def forward(self, x):
        x = self.fc(x)
        return x


class ClassBlock(nn.Module):
    def __init__(self, input_dim=512, class_num=751):
        super(ClassBlock, self).__init__()
        classifier = []
        classifier += [nn.Linear(input_dim, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)
        self.classifier = classifier

    def forward(self, x):
        x = self.classifier(x)
        return x


class BN(nn.Module):
    def __init__(self, input_dim=512):
        super(BN, self).__init__()
        bn = []
        bn += [nn.BatchNorm1d(input_dim)]
        bn = nn.Sequential(*bn)
        bn.apply(weights_init_kaiming)
        self.bn = bn

    def forward(self, x):
        x = self.bn(x)
        return x

# Define the ResNet50-based Model
class ft_net(nn.Module):

    def __init__(self, class_num):
        super(ft_net, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.model = model_ft
        self.classifier = Fc_ClassBlock(2048, class_num, dropout=0.5, relu=False)
        # remove the final downsample
        # self.model.layer4[0].downsample[0].stride = (1,1)
        # self.model.layer4[0].conv2.stride = (1,1)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        x, f = self.classifier(x)
        return x, f


# Define a 2048 to 2 Model
class verif_net(nn.Module):
    def __init__(self):
        super(verif_net, self).__init__()
        self.classifier = Fc_ClassBlock(512, 2, dropout=0.75, relu=False)

    def forward(self, x):
        x = self.classifier.classifier(x)
        return x


# Define the DenseNet121-based Model
class ft_net_dense(nn.Module):

    def __init__(self, class_num=751):
        super().__init__()
        model_ft = models.densenet121(pretrained=True)
        model_ft.features.avgpool = nn.Sequential()
        model_ft.fc = nn.Sequential()
        self.model = model_ft
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        # For DenseNet, the feature dim is 1024
        self.classifier = Fc_ClassBlock(1024, class_num)

    def forward(self, x):
        x = self.model.features(x)
        feature_space = x
        x = self.pool(x)
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x[0], x[1], feature_space


# Define the ResNet50-based Model (Middle-Concat)
# In the spirit of "The Devil is in the Middle: Exploiting Mid-level Representations for Cross-Domain Instance Matching." Yu, Qian, et al. arXiv:1711.08106 (2017).
class ft_net_middle(nn.Module):

    def __init__(self, class_num):
        super(ft_net_middle, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.model = model_ft
        self.classifier = Fc_ClassBlock(2048 + 1024, class_num)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        # x0  n*1024*1*1
        x0 = self.model.avgpool(x)
        x = self.model.layer4(x)
        # x1  n*2048*1*1
        x1 = self.model.avgpool(x)
        x = torch.cat((x0, x1), 1)
        x = torch.squeeze(x)
        x = self.classifier(x)
        return x


# Part Model proposed in Yifan Sun etal. (2018)
class PCB(nn.Module):
    def __init__(self, class_num):
        super(PCB, self).__init__()

        self.part = 6  # We cut the pool5 to 6 parts
        model_ft = models.resnet50(pretrained=True)
        self.model = model_ft
        self.avgpool = nn.AdaptiveAvgPool2d((self.part, 1))
        self.dropout = nn.Dropout(p=0.5)
        # remove the final downsample
        self.model.layer4[0].downsample[0].stride = (1, 1)
        self.model.layer4[0].conv2.stride = (1, 1)
        # define 6 classifiers
        for i in range(self.part):
            name = 'classifier' + str(i)
            setattr(self, name, Fc_ClassBlock(2048, class_num, True, False, 256))

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.avgpool(x)
        x = self.dropout(x)
        part = {}
        predict = {}
        # get six part feature batchsize*2048*6
        for i in range(self.part):
            part[i] = torch.squeeze(x[:, :, i])
            name = 'classifier' + str(i)
            c = getattr(self, name)
            predict[i] = c(part[i])

        # sum prediction
        # y = predict[0]
        # for i in range(self.part-1):
        #    y += predict[i+1]
        y = []
        for i in range(self.part):
            y.append(predict[i])
        return y


class PCB_test(nn.Module):
    def __init__(self, model):
        super(PCB_test, self).__init__()
        self.part = 6
        self.model = model.model
        self.avgpool = nn.AdaptiveAvgPool2d((self.part, 1))
        # remove the final downsample
        self.model.layer4[0].downsample[0].stride = (1, 1)
        self.model.layer4[0].conv2.stride = (1, 1)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.avgpool(x)
        y = x.view(x.size(0), x.size(1), x.size(2))
        return y



class SiameseNet(nn.Module):
    def __init__(self, embedding_net):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net
        self.classifier = Fc_ClassBlock(1024, 2, dropout=0.75, relu=False)
        # self.bn = BN(512)

    # def forward(self, x1, x2=None):
    #     output1, feature1 = self.embedding_net(x1)
    #     if x2 is None:
    #         return output1, feature1
    #     output2, feature2 = self.embedding_net(x2)
    #     feature = (feature1 - feature2).pow(2)
    #
    #     # f_norm = feature.norm(p=2, dim=1, keepdim=True) + 1e-8
    #     # feature = feature.div(f_norm)
    #
    #     # feature = self.bn(feature)
    #
    #     result = self.classifier.classifier(feature)
    #     return output1, feature1, output2, feature2, feature, result


    # def forward(self, x1, x2=None):
    #     output1, feature1 = self.embedding_net(x1)
    #     if x2 is None:
    #         return output1, feature1
    #     output2, feature2 = self.embedding_net(x2)
    #     feature_combine = torch.cat(
    #         (feature1[:, :int(feature1.size(1) / 2)], feature2[:, int(feature2.size(1) / 2):]), 1)
    #     f_norm = feature_combine.norm(p=2, dim=1, keepdim=True) + 1e-8
    #     feature_combine = feature_combine.div(f_norm)
    #     feature = (feature1 - feature2).pow(2)
    #     feature_combine_1 = (feature1 - feature_combine).pow(2)
    #     feature_combine_2 = (feature2 - feature_combine).pow(2)
    #     result = self.classifier.classifier(feature)
    #     result_combine_1 = self.classifier.classifier(feature_combine_1)
    #     result_combine_2 = self.classifier.classifier(feature_combine_2)
    #     return output1, feature1, feature1, output2, feature2, feature2, \
    #            result, result_combine_1, result_combine_2

    def forward(self, x1, x2=None):
        output1, feature1, feature_space1 = self.embedding_net(x1)
        if x2 is None:
            return output1, feature1
        output2, feature2, feature_space2 = self.embedding_net(x2)
        feature_combine = torch.cat(
            (feature_space1[:, :, :int(feature_space1.size(2)/2)],
             feature_space2[:, :, int(feature_space2.size(2)/2):]), 2)

        feature_combine = self.reduce_dim(feature_combine)
        feature_space1 = self.reduce_dim(feature_space1)
        feature_space2 = self.reduce_dim(feature_space2)

        feature = (feature_space1 - feature_space2).pow(2)
        feature_combine_1 = (feature_space1 - feature_combine).pow(2)
        feature_combine_2 = (feature_space2 - feature_combine).pow(2)
        result = self.classifier(feature)[0]
        result_combine_1 = self.classifier(feature_combine_1)[0]
        result_combine_2 = self.classifier(feature_combine_2)[0]
        return output1, output2, result, result_combine_1, result_combine_2

    def reduce_dim(self, x):
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), x.size(1))
        f_norm = x.norm(p=2, dim=1, keepdim=True) + 1e-8
        x = x.div(f_norm)
        return x

    def get_embedding(self, x):
        return self.embedding_net(x)


