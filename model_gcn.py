import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp


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
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

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
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
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

        pair_num = 16
        not_pair_num = num_g_per_batch - pair_num
        index = np.arange(num_p_per_batch)
        np.random.shuffle(index[pair_num:])
        x_g = x_g[index]
        if y is not None:
            y_g = y_g[index]

        index = np.random.permutation(num_p_per_batch)
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
        if y is not None:
            label = torch.where(y_p == y_g, torch.full_like(label, 1), torch.full_like(label, 0))
        for i in range(num_p_per_batch):
            adj[i, :] = (feature[i].unsqueeze(0).repeat(num_p_per_batch, 1) - feature).pow(2).sum(-1)
        adj = (-adj).exp()
        adj = self.preprocess_adj(adj)

        return adj, feature, label

    def preprocess_adj(self, adj):
        """Symmetrically normalize adjacency matrix."""
        adj = adj + torch.eye(adj.shape[0]).cuda()
        rowsum = torch.Tensor(adj.sum(1).cpu()).cuda()
        d_inv_sqrt = torch.pow(rowsum, -0.5).flatten()
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        return adj.mm(d_mat_inv_sqrt).transpose(0, 1).mm(d_mat_inv_sqrt)


