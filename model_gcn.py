import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
import pandas as pd
# import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.io import savemat
import time
import random
import os


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def load_data(path="../data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def load_data_reid(node_unabled, guider_path='nodes_info.mat'):
    guider_num = 1000
    m = loadmat(guider_path)
    # print('type(m) = %s' % type(m))
    # print(m.keys())
    node_same = m['feature_same']
    dist_same = node_same.sum(-1)
    node_dif = m['feature_dif']
    dist_dif = node_dif.sum(-1)
    dist_unabled = node_unabled.sum(-1)
    min_unabled = dist_unabled.min()
    max_unabled = dist_unabled.max()

    # #very bad
    # node_same = node_same[-guider_num:]
    # node_dif = node_dif[:guider_num]
    # #large improve
    # random_index = np.random.choice(np.arange(int(len(node_same) / 2), len(node_same)), guider_num, replace=False)
    # node_same = node_same[random_index]
    # random_index = np.random.choice(np.arange(int(len(node_dif) / 100)), guider_num, replace=False)
    # node_dif = node_dif[random_index]
    gcn_features = np.concatenate((node_same, node_dif, node_unabled), 0)
    same_num = len(node_same)
    dif_num = len(node_dif)
    total_num = len(gcn_features)
    labeled_num = same_num + dif_num
    unlabeled_num = len(node_unabled)
    labels = np.concatenate((np.ones(same_num), np.zeros(dif_num), np.ones(unlabeled_num)), 0)
    adj_m = np.zeros(shape=(total_num, total_num), dtype=np.float32)
    for i in range(total_num):
        for j in range(i):
            dif_feature = gcn_features[i] - gcn_features[j]
            temp_adj_value = dif_feature ** 2
            distance = np.sum(temp_adj_value)
            adj_m[i][j] = distance
            adj_m[j][i] = adj_m[i][j]

    # joint_num = 30
    # sparse_num_real = min(int(joint_num * 0.5), unlabeled_num)
    # sparse_num_gen = min(joint_num - sparse_num_real, unlabeled_num)
    # # print('sparse_num_real = %s   sparse_num_gen = %s' % (sparse_num_real, sparse_num_gen))
    #
    # for i in range(total_num):
    #     arr_index_real = np.argsort(adj_m[i][: labeled_num])
    #     arr_index_gen = np.argsort(adj_m[i][labeled_num:]) + labeled_num
    #     adj_m[i][arr_index_real[sparse_num_real:]] = 0  # 101 elements not zero(include itself)
    #     adj_m[i][arr_index_gen[sparse_num_gen:]] = 0  # 101 elements not zero(include itself)
    #     adj_m[i][arr_index_real[:sparse_num_real]] = np.exp(-adj_m[i][arr_index_real[:sparse_num_real]])
    #     adj_m[i][arr_index_gen[:sparse_num_gen]] = np.exp(-adj_m[i][arr_index_gen[:sparse_num_gen]])

    joint_num = 30
    sparse_num_same = min(joint_num, same_num)
    sparse_num_dif = min(joint_num, dif_num)
    sparse_num_unabled = min(joint_num, unlabeled_num)

    for i in range(total_num):
        arr_index_same = np.argsort(adj_m[i][: same_num])
        arr_index_dif = np.argsort(adj_m[i][same_num:same_num + dif_num]) + same_num
        arr_index_unabled = np.argsort(adj_m[i][labeled_num:]) + labeled_num
        adj_m[i][arr_index_same[sparse_num_same:]] = 0  # 101 elements not zero(include itself)
        adj_m[i][arr_index_dif[sparse_num_dif:]] = 0  # 101 elements not zero(include itself)
        adj_m[i][arr_index_unabled[sparse_num_unabled:]] = 0  # 101 elements not zero(include itself)
        adj_m[i][arr_index_same[:sparse_num_same]] = np.exp(-adj_m[i][arr_index_same[:sparse_num_same]])
        adj_m[i][arr_index_dif[:sparse_num_dif]] = np.exp(-adj_m[i][arr_index_dif[:sparse_num_dif]])
        adj_m[i][arr_index_unabled[:sparse_num_unabled]] = np.exp(-adj_m[i][arr_index_unabled[:sparse_num_unabled]])

    for i in range(total_num):
        for j in range(total_num):
            if math.fabs(adj_m[i][j]) > 1e-09:
                adj_m[j][i] = adj_m[i][j]
            elif math.fabs(adj_m[j][i]) > 1e-09:
                adj_m[i][j] = adj_m[j][i]
    # adj_m = np.exp(-adj_m)
    adj = sp.coo_matrix(adj_m)
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # gcn_features = normalize(gcn_features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    train_ratio = 0.8
    val_ratio = 1 - train_ratio
    idx_train = np.concatenate(
        (np.arange(0, int(same_num * train_ratio)), np.arange(same_num, same_num + dif_num * train_ratio)))
    idx_val = np.concatenate((np.arange(int(same_num * train_ratio), same_num),
                              np.arange(same_num + dif_num * train_ratio, same_num + dif_num)))
    idx_test = np.arange(labeled_num, total_num)

    gcn_features = torch.FloatTensor(np.array(gcn_features))
    labels = torch.LongTensor(labels)
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, gcn_features, labels, idx_train, idx_val, idx_test


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.detach().uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.detach().uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.gc3 = GraphConvolution(nhid, nhid)
        self.gc4 = GraphConvolution(nhid, nhid)
        self.gc5 = GraphConvolution(nhid, nhid)
        self.gc6 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu6(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        # x = F.relu(self.gc2(x, adj))
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = F.relu6(self.gc3(x, adj))
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = F.relu6(self.gc4(x, adj))
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = F.relu6(self.gc5(x, adj))
        # x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc6(x, adj)
        return F.log_softmax(x, dim=1)


class Sggnn_prepare(nn.Module):
    def __init__(self, siamesemodel, hard_weight=True):
        super(Sggnn_prepare, self).__init__()
        self.basemodel = siamesemodel
        self.hard_weight = hard_weight

    def forward(self, x, y=None):
        use_gpu = torch.cuda.is_available()
        batch_size = len(x)
        x_p = x[:, 0]
        x_g = x[:, 1]
        if y is not None:
            y_p = y[:, 0]
            y_g = y[:, 1]
        num_p_per_batch = len(x_p)  # 48
        num_g_per_batch = len(x_g)  # 48

        pair_num = np.random.randint(int(num_p_per_batch * 0.4), int(num_p_per_batch * 0.6))
        not_pair_num = num_g_per_batch - pair_num
        index = np.arange(num_p_per_batch)
        np.random.shuffle(index[pair_num:])
        x_g = x_g[index]
        if y is not None:
            y_g = y_g[index]

        index = np.random.permutation(num_p_per_batch)
        # index = np.arange(num_p_per_batch)
        x_p = x_p[index]
        x_g = x_g[index]
        if y is not None:
            y_p = y_p[index]
            y_g = y_g[index]

        len_feature = 512
        feature = torch.FloatTensor(num_p_per_batch, len_feature).zero_()
        label = torch.LongTensor(num_p_per_batch).zero_()
        adj = torch.FloatTensor(num_p_per_batch, num_g_per_batch).zero_()
        if use_gpu:
            feature = feature.cuda()
            label = label.cuda()
            adj = adj.cuda()

        feature = self.basemodel(x_p, x_g)[-2]
        feature = self.normalize(feature.cpu().numpy())
        feature = torch.from_numpy(feature).cuda().float()
        if y is not None:
            label = torch.where(y_p == y_g, torch.full_like(label, 1), torch.full_like(label, 0))
        for i in range(num_p_per_batch):
            adj[i, :] = (feature[i].unsqueeze(0).repeat(num_p_per_batch, 1) - feature).pow(2).sum(-1)
        # adj = (-adj).exp()
        # adj = self.preprocess_adj(adj)
        # #or
        # build symmetric adjacency matrix
        adj_np = adj.cpu().numpy()
        adj_np = adj_np + np.multiply(adj_np.T, adj_np.T > adj_np) - np.multiply(adj_np, adj_np.T > adj_np)
        adj_np = self.normalize(adj_np + sp.eye(adj_np.shape[0]))
        adj = torch.from_numpy(adj_np).cuda().float()

        # adj = torch.FloatTensor(num_p_per_batch, num_g_per_batch).fill_(
        #     1.0 / (num_p_per_batch * num_g_per_batch)).cuda()

        return adj, feature, label

    def preprocess_adj(self, adj):
        """Symmetrically normalize adjacency matrix."""
        adj = adj + torch.eye(adj.shape[0]).cuda()
        rowsum = torch.Tensor(adj.sum(1).cpu()).cuda()
        d_inv_sqrt = torch.pow(rowsum, -0.5).flatten()
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        return adj.mm(d_mat_inv_sqrt).transpose(0, 1).mm(d_mat_inv_sqrt)

    def normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx


class Sggnn_prepare_test(nn.Module):
    def __init__(self):
        super(Sggnn_prepare_test, self).__init__()

    def forward(self, qf, gf):
        use_gpu = torch.cuda.is_available()

        num_p_per_batch = len(gf)  # 128
        num_g_per_batch = len(gf)  # 128

        len_feature = 512
        feature = torch.FloatTensor(num_p_per_batch, len_feature).zero_()
        adj = torch.FloatTensor(num_p_per_batch, num_g_per_batch).zero_()
        if use_gpu:
            feature = feature.cuda()
            adj = adj.cuda()

        feature = (qf - gf).pow(2)
        feature = self.normalize(feature.cpu().numpy())
        feature = torch.from_numpy(feature).cuda().float()
        for i in range(num_p_per_batch):
            adj[i, :] = (feature[i].unsqueeze(0).repeat(num_p_per_batch, 1) - feature).pow(2).sum(-1)
        # adj = (-adj).exp()
        # adj = self.preprocess_adj(adj)
        # #or
        # build symmetric adjacency matrix
        adj_np = adj.cpu().numpy()
        adj_np = adj_np + np.multiply(adj_np.T, adj_np.T > adj_np) - np.multiply(adj_np, adj_np.T > adj_np)
        adj_np = self.normalize(adj_np + sp.eye(adj_np.shape[0]))
        adj = torch.from_numpy(adj_np).cuda().float()

        return adj, feature

    def preprocess_adj(self, adj):
        """Symmetrically normalize adjacency matrix."""
        adj = adj + torch.eye(adj.shape[0]).cuda()
        rowsum = torch.Tensor(adj.sum(1).cpu()).cuda()
        d_inv_sqrt = torch.pow(rowsum, -0.5).flatten()
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        return adj.mm(d_mat_inv_sqrt).transpose(0, 1).mm(d_mat_inv_sqrt)

    def normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx


class Random_walk(nn.Module):
    def __init__(self):
        super(Random_walk, self).__init__()

    def forward(self, qf, gf):
        use_gpu = torch.cuda.is_available()
        batch_size = len(qf)
        num_g_per_id = len(gf[0])  # 100
        d = (qf.unsqueeze(1) - gf).pow(2)
        d_new = torch.FloatTensor(d.shape).zero_()
        w = torch.FloatTensor(batch_size, num_g_per_id, num_g_per_id).zero_()
        if use_gpu:
            d = d.cuda()
            d_new = d_new.cuda()
            w = w.cuda()
        for i in range(num_g_per_id):
            for j in range(num_g_per_id):
                w[:, i, j] = (gf[:, i] - gf[:, j]).pow(2).sum(-1)
                w[:, i, j] = (-w[:, i, j]).exp()
        ratio = 0.95
        for i in range(batch_size):
            # w[i] = self.preprocess_adj(w[i])
            w[i] = self.preprocess_sggnn_adj(w[i])
            for j in range(d.shape[-1]):
                d_new[i, :, j] = torch.mm(d[i, :, j].unsqueeze(0), w[i])
                d_new[i, :, j] = ratio * d_new[i, :, j] + (1 - ratio) * d[i, :, j]
        # 1 for similar & 0 for different
        # d_new is different from (pf - gf).pow(2)
        result = d_new.pow(2).sum(-1)
        result = (-result).exp()
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
