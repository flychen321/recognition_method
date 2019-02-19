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

# class Sggnn_all(nn.Module):
#     def __init__(self, siamesemodel, hard_weight=True):
#         super(Sggnn_all, self).__init__()
#         self.basemodel = siamesemodel
#         self.rf = ReFineBlock(layer=2)
#         self.classifier = Fc_ClassBlock(input_dim=512, class_num=2, dropout=0.75, relu=False)
#         self.hard_weight = hard_weight
#
#     def forward(self, x, y=None):
#         use_gpu = torch.cuda.is_available()
#         batch_size = len(x)
#         x_p = x[:, 0]
#         x_g = x[:, 1:]
#         num_p_per_batch = len(x_p)  # 48
#         num_g_per_batch = len(x_g) * len(x_g[0])  # 144
#         len_feature = 512
#         d = torch.FloatTensor(num_p_per_batch, num_g_per_batch, len_feature).zero_()
#         t = torch.FloatTensor(d.shape).zero_()
#         d_new = torch.FloatTensor(d.shape).zero_()
#         result = torch.FloatTensor(d.shape[: -1] + (2,)).zero_()
#         # this w for dynamic calculate the weight
#         # this w for calculate the weight by label too
#         w = torch.FloatTensor(num_g_per_batch, num_g_per_batch).zero_()
#         label = torch.LongTensor(num_p_per_batch, num_g_per_batch).zero_()
#
#         if use_gpu:
#             d = d.cuda()
#             t = t.cuda()
#             d_new = d_new.cuda()
#             w = w.cuda()
#             label = label.cuda()
#         if y is not None:
#             y_p = y[:, 0]
#             y_g = y[:, 1:]
#
#         x_g = x_g.reshape((-1,) + x_g.shape[2:])
#         y_g = y_g.reshape((-1,) + y_g.shape[2:])
#
#         for i in range(num_p_per_batch):
#             d[i, :] = self.basemodel(x_p[i].unsqueeze(0).repeat(len(x_g), 1, 1, 1), x_g)[-2]
#             if y is not None:
#                 label[i, :] = torch.where(y_p[i].unsqueeze(0) == y_g, torch.full_like(label[i, :], 1),
#                                           torch.full_like(label[i, :], 0))
#         for i in range(num_g_per_batch):
#             if self.hard_weight and y is not None:
#                 w[i, :] = torch.where(y_g[i].unsqueeze(0) == y_g, torch.full_like(w[i, :], 1),
#                                       torch.full_like(w[i, :], 0))
#             else:
#                 # model output 1 for similar & 0 for different
#                 w[i, :] = F.softmax(self.basemodel(x_g[i].unsqueeze(0).repeat(len(x_g), 1, 1, 1), x_g)[-1], -1)[:, 0]
#                 # w[i, :] = self.basemodel(x_g[i].unsqueeze(0), x_g)[-2]
#
#         for i in range(num_p_per_batch):
#             t[i, :] = self.rf(d[i, :])
#
#         # w need to be normalized
#         # w = w - w.diag().diag().cuda()
#         w = self.preprocess_adj(w)
#         for i in range(t.shape[-1]):
#             d_new[:, :, i] = torch.mm(t[:, :, i], w)
#
#         # maybe need to fix
#         for i in range(num_p_per_batch):
#             feature = self.classifier.classifier(d_new[i, :])
#             result[i, :] = feature.squeeze()
#
#         result = result.view((num_p_per_batch * num_g_per_batch), -1)
#         if label is not None:
#             label = label.view(label.size(0) * label.size(1))
#
#         # print('run Sggnn_gcn foward success  !!!')
#         if label is not None:
#             return result, label
#         else:
#             return result
#
#     def normalize(self, mx):
#         """Row-normalize sparse matrix"""
#         rowsum = np.array(mx.sum(1))
#         r_inv = np.power(rowsum, -1).flatten()
#         r_inv[np.isinf(r_inv)] = 0.
#         r_mat_inv = sp.diags(r_inv)
#         mx = r_mat_inv.dot(mx)
#         return mx
#
#     def preprocess_features(self, features):
#         """Row-normalize feature matrix and convert to tuple representation"""
#         rowsum = np.array(features.sum(1))
#         r_inv = np.power(rowsum, -1).flatten()
#         r_inv[np.isinf(r_inv)] = 0.
#         r_mat_inv = sp.diags(r_inv)
#         features = r_mat_inv.dot(features)
#         return features
#
#     def preprocess_adj(self, adj):
#         """Symmetrically normalize adjacency matrix."""
#         adj = adj + torch.eye(adj.shape[0]).cuda()
#         rowsum = torch.Tensor(adj.sum(1).cpu()).cuda()
#         d_inv_sqrt = torch.pow(rowsum, -0.5).flatten()
#         d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
#         d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
#         return adj.mm(d_mat_inv_sqrt).transpose(0, 1).mm(d_mat_inv_sqrt)

a = torch.Tensor([[1,2,3], [1,2,8]])
print(a)
print(F.softmax(a, 0))
print(F.softmax(a, 0)[:,0].sum())