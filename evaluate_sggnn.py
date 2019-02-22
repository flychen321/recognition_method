import scipy.io
import torch
import numpy as np
import time
import os
import matplotlib
import torch.nn.functional as F
matplotlib.use('agg')
import matplotlib.pyplot as plt
from model_siamese import Sggnn_siamese, Sggnn_gcn, Sggnn_for_test
from model_siamese import load_network_easy, load_network, save_network, save_whole_network
from model_gcn import GCN, Sggnn_prepare, Sggnn_prepare_test, Random_walk
from model_siamese import ft_net_dense, SiameseNet
######################################################################
# Trained model
print('-------evaluate-----------')
name = 'sggnn'
use_gpu = torch.cuda.is_available()

embedding_net = ft_net_dense(751)
# embedding_net = ft_net_dense(702)
model_siamese = SiameseNet(embedding_net)
model_siamese = load_network_easy(model_siamese, name)
model_siamese = model_siamese.eval()
model_gcn = Random_walk(model_siamese)
if use_gpu:
    model = model_gcn.cuda()

cam_metric = torch.zeros(6, 6)
# cam_metric = torch.zeros(8, 8)


def evaluate(qf, ql, qc, gf, gl, gc, model=model):
    model.eval()
    batchsize = len(qf)
    index = np.zeros((batchsize, len(gf)), dtype=np.int32)
    ap_tmp = np.zeros((batchsize,), dtype=np.float)
    CMC_tmp = torch.IntTensor(batchsize, len(gf)).zero_()
    for i in range(batchsize):
        score = ((gf - qf[i]).pow(2)).sum(1)  # Ed distance
        score = score.cpu().numpy()
        # predict index
        index[i] = np.argsort(score)  # from small to large

    # operate for sggnn
    g_num = 100
    with torch.no_grad():
        index_new_100 = model(qf, gf[index[:, :g_num]])
        for i in range(batchsize):
            index[i, :g_num] = index[i, :g_num][index_new_100[i]]

    for i in range(batchsize):
        # good index
        query_index = np.argwhere(gl == ql[i])
        # same camera
        camera_index = np.argwhere(gc == qc[i])

        good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
        junk_index1 = np.argwhere(gl == -1)
        junk_index2 = np.intersect1d(query_index, camera_index)
        junk_index = np.append(junk_index2, junk_index1)  # .flatten())

        ap_tmp[i], CMC_tmp[i] = compute_mAP(index[i], qc, good_index, junk_index)

    return ap_tmp, CMC_tmp


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

query_feature = query_feature.cuda()
gallery_feature = gallery_feature.cuda()

print(query_feature.shape)
CMC = torch.IntTensor(len(gallery_label)).zero_()
ap = 0.0
# print(query_label)

batchsize = 256
right_cnt = 0
i = 0
former_right_cnt = 0
former_i = 0
while i < len(query_label):
    ap_tmp, CMC_tmp = evaluate(query_feature[i: min(i + batchsize, len(query_label))],
                               query_label[i: min(i + batchsize, len(query_label))],
                               query_cam[i: min(i + batchsize, len(query_label))], gallery_feature, gallery_label,
                               gallery_cam)

    for j in range(min(batchsize, len(query_label) - i)):
        if CMC_tmp[j][0] == -1:
            continue
        CMC = CMC + CMC_tmp[j]
        ap += ap_tmp[j]

        if CMC_tmp[j][0].numpy() == 1:
            right_cnt += 1

    i += min(batchsize, len(query_label) - i)
    print('i = %4d    CMC_tmp[0] = %s  real-time rank1 = %.4f  avg rank1 = %.4f' % (
        i, CMC_tmp[0].numpy(), float(right_cnt - former_right_cnt) / (i - former_i), float(right_cnt) / (i)))
    former_right_cnt = right_cnt
    former_i = i

CMC = CMC.float()
CMC = CMC / len(query_label)  # average CMC
print('Rank@1:%.4f  Rank@2:%.4f  Rank@5:%.4f  Rank@10:%.4f  mAP:%.4f' % (
CMC[0], CMC[1], CMC[4], CMC[9], ap / len(query_label)))

# multiple-query
CMC = torch.IntTensor(len(gallery_label)).zero_()
ap = 0.0
