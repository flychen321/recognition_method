import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable
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
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)


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
        model_ft.features.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        model_ft.fc = nn.Sequential()
        self.model = model_ft
        # For DenseNet, the feature dim is 1024 
        self.classifier = Fc_ClassBlock(1024, class_num)

    def forward(self, x):
        x = self.model.features(x)
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x


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


# debug model structure
# net = ft_net(751)
# net = ft_net(751)
# print(net)
# input = Variable(torch.FloatTensor(8, 3, 224, 224))
# output,f = net(input)
# print('net output size:')
# print(f.shape)

class SiameseNet(nn.Module):
    def __init__(self, embedding_net):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net
        self.classifier = Fc_ClassBlock(512, 2, dropout=0.75, relu=False)
        # self.bn = BN(512)

    def forward(self, x1, x2=None):
        output1, feature1 = self.embedding_net(x1)
        if x2 is None:
            return output1, feature1
        output2, feature2 = self.embedding_net(x2)
        feature = (feature1 - feature2).pow(2)

        # f_norm = feature.norm(p=2, dim=1, keepdim=True) + 1e-8
        # feature = feature.div(f_norm)

        # feature = self.bn(feature)

        result = self.classifier.classifier(feature)
        return output1, feature1, output2, feature2, feature, result


    def get_embedding(self, x):
        return self.embedding_net(x)


# There maybe some problems in sggnn_all
class Sggnn_all(nn.Module):
    def __init__(self, siamesemodel, hard_weight=True):
        super(Sggnn_all, self).__init__()
        self.basemodel = siamesemodel
        self.rf = ReFineBlock(layer=2)
        self.classifier = Fc_ClassBlock(input_dim=512, class_num=2, dropout=0.75, relu=False)
        self.hard_weight = hard_weight

    def forward(self, x, y=None):
        use_gpu = torch.cuda.is_available()
        batch_size = len(x)
        x_p = x[:, 0]
        x_g = x[:, 1:]
        num_p_per_batch = len(x_p)  # 48
        num_g_per_batch = len(x_g) * len(x_g[0])  # 144
        len_feature = 512
        d = torch.FloatTensor(num_p_per_batch, num_g_per_batch, len_feature).zero_()
        t = torch.FloatTensor(d.shape).zero_()
        d_new = torch.FloatTensor(d.shape).zero_()
        result = torch.FloatTensor(d.shape[: -1] + (2,)).zero_()
        # this w for dynamic calculate the weight
        # this w for calculate the weight by label too
        w = torch.FloatTensor(num_g_per_batch, num_g_per_batch).zero_()
        label = torch.LongTensor(num_p_per_batch, num_g_per_batch).zero_()

        if use_gpu:
            d = d.cuda()
            t = t.cuda()
            d_new = d_new.cuda()
            w = w.cuda()
            label = label.cuda()
        if y is not None:
            y_p = y[:, 0]
            y_g = y[:, 1:]

        x_g = x_g.reshape((-1,) + x_g.shape[2:])
        y_g = y_g.reshape((-1,) + y_g.shape[2:])

        for i in range(num_p_per_batch):
            d[i, :] = self.basemodel(x_p[i].unsqueeze(0).repeat(len(x_g), 1, 1, 1), x_g)[-2]
            if y is not None:
                label[i, :] = torch.where(y_p[i].unsqueeze(0) == y_g, torch.full_like(label[i, :], 1),
                                          torch.full_like(label[i, :], 0))
        for i in range(num_g_per_batch):
            if self.hard_weight and y is not None:
                w[i, :] = torch.where(y_g[i].unsqueeze(0) == y_g, torch.full_like(w[i, :], 1),
                                      torch.full_like(w[i, :], 0))
            else:
                # model output 1 for similar & 0 for different
                w[i, :] = F.softmax(self.basemodel(x_g[i].unsqueeze(0).repeat(len(x_g), 1, 1, 1), x_g)[-1], -1)[:, 0]
                # w[i, :] = self.basemodel(x_g[i].unsqueeze(0), x_g)[-2]

        for i in range(num_p_per_batch):
            t[i, :] = self.rf(d[i, :])

        # w need to be normalized
        # w = w - w.diag().diag().cuda()
        w = self.preprocess_adj(w)
        for i in range(t.shape[-1]):
            d_new[:, :, i] = torch.mm(t[:, :, i], w)

        # maybe need to fix
        for i in range(num_p_per_batch):
            feature = self.classifier.classifier(d_new[i, :])
            result[i, :] = feature.squeeze()

        result = result.view((num_p_per_batch * num_g_per_batch), -1)
        if label is not None:
            label = label.view(label.size(0) * label.size(1))

        # print('run Sggnn_gcn foward success  !!!')
        if label is not None:
            return result, label
        else:
            return result

    def normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

    def preprocess_features(self, features):
        """Row-normalize feature matrix and convert to tuple representation"""
        rowsum = np.array(features.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        features = r_mat_inv.dot(features)
        return features

    def preprocess_adj(self, adj):
        """Symmetrically normalize adjacency matrix."""
        adj = adj + torch.eye(adj.shape[0]).cuda()
        rowsum = torch.Tensor(adj.sum(1).cpu()).cuda()
        d_inv_sqrt = torch.pow(rowsum, -0.5).flatten()
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        return adj.mm(d_mat_inv_sqrt).transpose(0, 1).mm(d_mat_inv_sqrt)

class Sggnn_siamese(nn.Module):
    def __init__(self, siamesemodel, hard_weight=True):
        super(Sggnn_siamese, self).__init__()
        self.basemodel = siamesemodel
        self.hard_weight = hard_weight

    def forward(self, x, y=None):
        use_gpu = torch.cuda.is_available()
        batch_size = len(x)
        x_p = x[:, 0]
        x_g = x[:, 1:]
        num_p_per_batch = len(x_p)  # 48
        num_g_per_batch = len(x_g) * len(x_g[0])  # 144
        len_feature = 512
        d = torch.FloatTensor(num_p_per_batch, num_g_per_batch, len_feature).zero_()
        # this w for dynamic calculate the weight
        # this w for calculate the weight by label too
        w = torch.FloatTensor(num_g_per_batch, num_g_per_batch).zero_()
        label = torch.LongTensor(num_p_per_batch, num_g_per_batch).zero_()

        if use_gpu:
            d = d.cuda()
            w = w.cuda()
            label = label.cuda()
        if y is not None:
            y_p = y[:, 0]
            y_g = y[:, 1:]

        x_g = x_g.reshape((-1,) + x_g.shape[2:])
        y_g = y_g.reshape((-1,) + y_g.shape[2:])

        for i in range(num_p_per_batch):
            d[i, :] = self.basemodel(x_p[i].unsqueeze(0).repeat(len(x_g), 1, 1, 1), x_g)[-2]
            if y is not None:
                label[i, :] = torch.where(y_p[i].unsqueeze(0) == y_g, torch.full_like(label[i, :], 1),
                                          torch.full_like(label[i, :], 0))
        for i in range(num_g_per_batch):
            if self.hard_weight and y is not None:
                w[i, :] = torch.where(y_g[i].unsqueeze(0) == y_g, torch.full_like(w[i, :], 3),
                                      torch.full_like(w[i, :], 0))
            else:
                # model output 1 for similar & 0 for different
                # w[i, :] = F.softmax(self.basemodel(x_g[i].unsqueeze(0).repeat(len(x_g), 1, 1, 1), x_g)[-1], -1)[:, -1]
                # #or
                w[i, :] = self.basemodel(x_g[i].unsqueeze(0), x_g)[-2].sum(-1)
                w[i, :] = (-w[i, :]).exp()
                # w[i, :] = w[i, :] * (-3.0)

        if y is not None:
            return d, w, label
        else:
            return d, w


class Sggnn_gcn(nn.Module):
    def __init__(self):
        super(Sggnn_gcn, self).__init__()
        self.rf = ReFineBlock(layer=2)
        self.classifier = Fc_ClassBlock(input_dim=512, class_num=2, dropout=0.75, relu=False)

    def forward(self, d, w, label=None):
        use_gpu = torch.cuda.is_available()
        batch_size = len(d)
        num_p_per_batch = len(d)  # 48
        num_g_per_batch = len(w)  # 144
        t = torch.FloatTensor(d.shape).zero_()
        d_new = torch.FloatTensor(d.shape).zero_()
        result = torch.FloatTensor(d.shape[: -1] + (2,)).zero_()

        if use_gpu:
            d = d.cuda()
            t = t.cuda()
            d_new = d_new.cuda()
            w = w.cuda()
            label = label.cuda()

        for i in range(num_p_per_batch):
            t[i, :] = self.rf(d[i, :])

        # w need to be normalized
        # w = self.preprocess_adj(w)
        w = self.preprocess_sggnn_adj(w)
        ratio = 0.9
        for i in range(t.shape[-1]):
            d_new[:, :, i] = torch.mm(t[:, :, i], w)
            d_new[:, :, i] = ratio * d_new[:, :, i] + (1 - ratio) * d[:, :, i]

        # maybe need to fix
        for i in range(num_p_per_batch):
            feature = self.classifier.classifier(d_new[i, :])
            result[i, :] = feature.squeeze()

        result = result.view((num_p_per_batch * num_g_per_batch), -1)
        if label is not None:
            label = label.view(label.size(0) * label.size(1))

        # print('run Sggnn_gcn foward success  !!!')
        if label is not None:
            return result, label
        else:
            return result

    def normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

    def preprocess_features(self, features):
        """Row-normalize feature matrix and convert to tuple representation"""
        rowsum = np.array(features.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        features = r_mat_inv.dot(features)
        return features

    def preprocess_adj(self, adj):
        """Symmetrically normalize adjacency matrix."""
        adj = adj + torch.eye(adj.shape[0]).cuda()
        rowsum = torch.Tensor(adj.sum(1).cpu()).cuda()
        d_inv_sqrt = torch.pow(rowsum, -0.5).flatten()
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        return adj.mm(d_mat_inv_sqrt).transpose(0, 1).mm(d_mat_inv_sqrt)

    def preprocess_sggnn_adj(self, adj):
        """normalize adjacency matrix."""
        adj = F.softmax((adj - 100 * adj.diag().diag()), 0)
        return adj

class Sggnn_for_test(nn.Module):
    def __init__(self):
        super(Sggnn_for_test, self).__init__()
        self.rf = ReFineBlock(layer=2)
        self.classifier = Fc_ClassBlock(input_dim=512, class_num=2, dropout=0.75, relu=False)

    def forward(self, qf, gf):
        use_gpu = torch.cuda.is_available()
        batch_size = len(qf)
        num_g_per_id = len(gf[0])  # 100
        d = (qf.unsqueeze(1) - gf).pow(2)
        t = torch.FloatTensor(d.shape).zero_()
        d_new = torch.FloatTensor(d.shape).zero_()
        w = torch.FloatTensor(batch_size, num_g_per_id, num_g_per_id).zero_()
        if use_gpu:
            d = d.cuda()
            d_new = d_new.cuda()
            t = t.cuda()
            w = w.cuda()
        for i in range(num_g_per_id):
            for j in range(num_g_per_id):
                w[:, i, j] = (gf[:, i] - gf[:, j]).pow(2).sum(-1)
                w[:, i, j] = (-w[:, i, j]).exp()
                # w[i, :] = w[i, :] * (-3.0)
                # #or
                # w[:, i, j] = F.softmax(self.classifier.classifier((gf[:, i] - gf[:, j]).pow(2)), -1)[:, -1]
        for i in range(num_g_per_id):
            t[:, i] = self.rf(d[:, i])
        ratio = 0.1
        for i in range(batch_size):
            # w[i] = self.preprocess_adj(w[i])
            w[i] = self.preprocess_sggnn_adj(w[i])
            for j in range(t.shape[-1]):
                # d_new[i, :, j] = t[i, :, j]
                # #or
                d_new[i, :, j] = torch.mm(t[i, :, j].unsqueeze(0), w[i])
                d_new[i, :, j] = ratio * d_new[i, :, j] + (1 - ratio) * d[i, :, j]
                # #or without d -> t
                # d_new[i, :, j] = torch.mm(d[i, :, j].unsqueeze(0), w[i])
                # d_new[i, :, j] = ratio * d_new[i, :, j] + (1 - ratio) * d[i, :, j]
        # d_new is different from (pf - gf).pow(2)
        # result = d_new.pow(2).sum(-1)
        # #result = (-result).exp()
        # w[i, :] = w[i, :] * (-3.0)
        # #or
        # 1 for similar & 0 for different
        result = F.softmax(self.classifier.classifier(d_new), -1)[:, :, -1]
        _, index = torch.sort(result, -1, descending=True)
        return index

    def preprocess_adj(self, adj):
        """Symmetrically normalize adjacency matrix."""
        adj = adj + torch.eye(adj.shape[0]).cuda()
        rowsum = torch.Tensor(adj.sum(1).cpu()).cuda()
        d_inv_sqrt = torch.pow(rowsum, -0.5).flatten()
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        return adj.mm(d_mat_inv_sqrt).transpose(0, 1).mm(d_mat_inv_sqrt)

    def preprocess_sggnn_adj(self, adj):
        """normalize adjacency matrix."""
        adj = F.softmax((adj - 100 * adj.diag().diag()), 0)
        return adj

