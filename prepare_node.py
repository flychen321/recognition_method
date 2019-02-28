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


def get_original_node():
    data_dir = 'data/market/pytorch/train_all'
    image_datasets = datasets.ImageFolder(data_dir)
    cams, labels = get_id(image_datasets.imgs)

    result = scipy.io.loadmat('pytorch_result.mat')
    query_feature = torch.FloatTensor(result['query_f'])
    query_cam = result['query_cam'][0]
    query_label = result['query_label'][0]
    gallery_feature = torch.FloatTensor(result['gallery_f'])
    gallery_cam = result['gallery_cam'][0]
    gallery_label = result['gallery_label'][0]

    train_feature = torch.FloatTensor(result['train_f'])
    train_cam = result['train_cam'][0]
    train_label = result['train_label'][0]
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
    print('len(node_same) = %d' % (len(node_same)))
    print('len(node_dif) = %d' % (len(node_dif)))

    result = {'feature_same': node_same.numpy(), 'feature_dif': node_dif.numpy(),
              'dist_same': dist_same.numpy(), 'dist_dif': dist_dif.numpy()}
    scipy.io.savemat('nodes_info_original.mat', result)
    return node_same, node_dif


def get_guider_node(node_same, node_dif):
    use_gpu = torch.cuda.is_available()
    node_dif = node_dif[node_dif.sum(-1).sort()[1][:int(len(node_dif)/2)]]
    cluster_num = min(2000, len(node_same))
    thre_num = int(len(node_same) / 1000)
    cluster_index = np.random.choice(np.arange(len(node_same)), cluster_num, replace=False)
    small_cluster_same = torch.Tensor(cluster_num, (thre_num - 1))
    if use_gpu:
        node_same = node_same.cuda()
        small_cluster_same = small_cluster_same.cuda()
    for i in np.arange(cluster_num):
        mid = (node_same[cluster_index[i]] - node_same).pow(2).sum(-1)
        small_cluster_same[i] = mid[mid.sort()[1][1:thre_num]]
    thre_same = small_cluster_same.mean()

    cluster_num = min(2000, len(node_dif))
    thre_num = int(len(node_dif) / 1000)
    cluster_index = np.random.choice(np.arange(len(node_dif)), cluster_num, replace=False)
    small_cluster_dif = torch.Tensor(cluster_num, (thre_num - 1))
    if use_gpu:
        node_dif = node_dif.cuda()
        small_cluster_dif = small_cluster_dif.cuda()
    for i in np.arange(cluster_num):
        mid = (node_dif[cluster_index[i]] - node_dif).pow(2).sum(-1)
        small_cluster_dif[i] = mid[mid.sort()[1][1:thre_num]]
    thre_dif = small_cluster_dif.mean()

    center_num = 1000
    iterate_num = center_num
    node_same_cluster = torch.Tensor(center_num*2, node_same.shape[-1])
    if use_gpu:
        node_same_cluster = node_same_cluster.cuda()
    i = 0
    j = 0
    num = []
    thre_same *= 2
    while i < iterate_num and node_same.shape[0] > 100:
        index = np.random.randint(len(node_same))
        # index = i % len(node_same)
        center = node_same[index]
        mid = (center - node_same).pow(2).sum(-1)
        if (mid < thre_same).sum() >= 3:
            # node_same_cluster[j] = center
            node_same_cluster[j] = node_same[mid < thre_same].mean(0)
            node_same = node_same[mid >= thre_same]
            # print((mid < thre_same).sum())
            num.append((mid < thre_same).sum())
            j += 1
        i += 1
    node_same_cluster = node_same_cluster[:j]
    print(len(num))
    print(sum(num))
    print(num)
    print(i, j, node_same.shape[0])
    center_num = 1000
    iterate_num = center_num
    node_dif_cluster = torch.Tensor(center_num, node_dif.shape[-1])
    if use_gpu:
        node_dif_cluster = node_dif_cluster.cuda()
    i = 0
    j = 0
    num = []
    thre_dif *= 2.0
    while i < iterate_num and node_dif.shape[0] > 100:
        index = np.random.randint(len(node_dif))
        # index = i % len(node_same)
        center = node_dif[index]
        mid = (center - node_dif).pow(2).sum(-1)
        if (mid < thre_dif).sum() >= 100:
            node_dif_cluster[j] = center
            node_dif = node_dif[mid >= thre_dif]
            # print((mid < thre_dif).sum())
            num.append((mid < thre_dif).sum())
            j += 1
        i += 1
    node_dif_cluster = node_dif_cluster[:j]
    print(len(num))
    print(sum(num))
    print(num)
    print(i, j, node_dif.shape[0])

    print('len(node_same_cluster) = %d' % (len(node_same_cluster)))
    print('len(node_dif_cluster) = %d' % (len(node_dif_cluster)))
    result = {'feature_same': node_same_cluster.cpu().numpy(), 'feature_dif': node_dif_cluster.cpu().numpy()}
    scipy.io.savemat('nodes_info.mat', result)

    return node_same_cluster, node_dif_cluster


if __name__ == '__main__':
    node_same, node_dif = get_original_node()
    get_guider_node(node_same, node_dif)
