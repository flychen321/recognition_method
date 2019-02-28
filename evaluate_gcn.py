import scipy.io
import torch
import numpy as np
import time
import os
import matplotlib
from model_gcn import load_data_reid
import torch.nn.functional as F
matplotlib.use('agg')
import matplotlib.pyplot as plt
from model_gcn import GCN, accuracy, Random_walk, random_walk_guider
from model_siamese import ft_net_dense, SiameseNet, load_network_easy
import torch.optim as optim
from torch.optim import lr_scheduler
#######################################################################
# Evaluate

# cam_metric = torch.zeros(6, 6)
cam_metric = torch.zeros(8, 8)

use_gpu = torch.cuda.is_available()

random_walk = Random_walk()

def label_propogate(unlabled_node):
    # Model and optimizer
    adj, features, labels, idx_train, idx_val, idx_test = load_data_reid(unlabled_node)
    model = GCN(nfeat=features.shape[1],
                nhid=64,
                nclass=labels.max().item() + 1,
                dropout=0.5)
    optimizer = optim.Adam(model.parameters(),
                           lr=0.01, weight_decay=5e-4)
    if use_gpu:
        model.cuda()
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()

    for epoch in range(50):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        output = model(features, adj)
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        acc_train = accuracy(output[idx_train], labels[idx_train])
        loss_train.backward()
        optimizer.step()

        loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        acc_val = accuracy(output[idx_val], labels[idx_val])
    #     if epoch % 5 == 0:
    #         print('Epoch: {:04d}'.format(epoch + 1),
    #               'loss_train: {:.4f}'.format(loss_train.item()),
    #               'acc_train: {:.4f}'.format(acc_train.item()),
    #               'loss_val: {:.4f}'.format(loss_val.item()),
    #               'acc_val: {:.4f}'.format(acc_val.item()),
    #               'time: {:.4f}s'.format(time.time() - t))
    # exit()
    index_new = F.softmax(output[idx_test], 0)[:, 0].sort()[1]
    return index_new


def evaluate(qf, ql, qc, gf, gl, gc):
    query = qf.view(1, -1)
    score = (gf - query).pow(2).sum(1)
    score = score.cpu().numpy()
    # predict index
    index = np.argsort(score)  # from small to large
    g_num = 100
    # #baseline:0  gcn:1  random_walk:2  guider:3
    mode = 2
    if mode == 1:
        unlabled_node = (gf[index[:g_num]] - query).pow(2)
        index_new_100 = label_propogate(unlabled_node)
        index[:g_num] = index[:g_num][index_new_100]
    elif mode == 2:
        index_new_100 = random_walk(qf.unsqueeze(0), gf[index[:g_num]].unsqueeze(0))
        index[:g_num] = index[:g_num][index_new_100]
    elif mode == 3:
        index_new_100 = random_walk_guider(qf, gf[index[:g_num]])
        index[:g_num] = index[:g_num][index_new_100]

    # good index
    query_index = np.argwhere(gl == ql)
    # same camera
    camera_index = np.argwhere(gc == qc)

    good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    junk_index1 = np.argwhere(gl == -1)
    junk_index2 = np.intersect1d(query_index, camera_index)
    junk_index = np.append(junk_index2, junk_index1)  # .flatten())

    CMC_tmp = compute_mAP(index, qc, good_index, junk_index)
    return CMC_tmp


def compute_mAP(index, qc, good_index, junk_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size == 0:  # if empty
        cmc[0] = -1
        return ap, cmc

    # remove junk_index
    ranked_camera = gallery_cam[index]
    mask = np.in1d(index, junk_index, invert=True)
    # mask2 = np.in1d(index, np.append(good_index,junk_index), invert=True)
    index = index[mask]
    ranked_camera = ranked_camera[mask]
    for i in range(10):
        cam_metric[qc - 1, ranked_camera[i] - 1] += 1

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask == True)
    rows_good = rows_good.flatten()

    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0 / ngood
        precision = (i + 1) * 1.0 / (rows_good[i] + 1)
        if rows_good[i] != 0:
            old_precision = i * 1.0 / rows_good[i]
        else:
            old_precision = 1.0
        ap = ap + d_recall * (old_precision + precision) / 2

    return ap, cmc


######################################################################
result = scipy.io.loadmat('pytorch_result.mat')
query_feature = torch.FloatTensor(result['query_f'])
query_cam = result['query_cam'][0]
query_label = result['query_label'][0]
gallery_feature = torch.FloatTensor(result['gallery_f'])
gallery_cam = result['gallery_cam'][0]
gallery_label = result['gallery_label'][0]

multi = os.path.isfile('multi_query.mat')

if multi:
    m_result = scipy.io.loadmat('multi_query.mat')
    mquery_feature = torch.FloatTensor(m_result['mquery_f'])
    mquery_cam = m_result['mquery_cam'][0]
    mquery_label = m_result['mquery_label'][0]
    mquery_feature = mquery_feature.cuda()

query_feature = query_feature.cuda()
gallery_feature = gallery_feature.cuda()

print(query_feature.shape)
CMC = torch.IntTensor(len(gallery_label)).zero_()
ap = 0.0
right_cnt = 0
former_right_cnt = 0
former_i = 0
# print(query_label)
# for i in range(len(query_label)):
for i in range(100):
    ap_tmp, CMC_tmp = evaluate(query_feature[i], query_label[i], query_cam[i], gallery_feature, gallery_label,
                               gallery_cam)
    if CMC_tmp[0] == -1:
        continue
    CMC = CMC + CMC_tmp
    ap += ap_tmp
    # print(i, CMC_tmp[0])

    if CMC_tmp[0].numpy() == 1:
        right_cnt += 1
    if i % 5 == 0 or i == len(query_label) - 1:
        print('i = %4d    CMC_tmp[0] = %s  real-time rank1 = %.4f  avg rank1 = %.4f' % (
        i, CMC_tmp[0].numpy(), float(right_cnt - former_right_cnt) / (i - former_i + 1), float(right_cnt) / (i + 1)))
        former_right_cnt = right_cnt
        former_i = i

CMC = CMC.float()
# CMC = CMC / len(query_label)  # average CMC
CMC = CMC / 100  # average CMC
# print('Rank@1:%f Rank@5:%f Rank@10:%f mAP:%f' % (CMC[0], CMC[4], CMC[9], ap / len(query_label)))
print('Rank@1:%f Rank@5:%f Rank@10:%f mAP:%f' % (CMC[0], CMC[4], CMC[9], ap / 100))

# multiple-query
CMC = torch.IntTensor(len(gallery_label)).zero_()
ap = 0.0
if multi:
    for i in range(len(query_label)):
        mquery_index1 = np.argwhere(mquery_label == query_label[i])
        mquery_index2 = np.argwhere(mquery_cam == query_cam[i])
        mquery_index = np.intersect1d(mquery_index1, mquery_index2)
        mq = torch.mean(mquery_feature[mquery_index, :], dim=0)
        ap_tmp, CMC_tmp = evaluate(mq, query_label[i], query_cam[i], gallery_feature, gallery_label, gallery_cam)
        if CMC_tmp[0] == -1:
            continue
        CMC = CMC + CMC_tmp
        ap += ap_tmp
        # print(i, CMC_tmp[0])
    CMC = CMC.float()
    CMC = CMC / len(query_label)  # average CMC
    print('multi Rank@1:%f Rank@5:%f Rank@10:%f mAP:%f' % (CMC[0], CMC[4], CMC[9], ap / len(query_label)))
