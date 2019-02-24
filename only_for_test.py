import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.io import savemat
import time
import random
import os
import math

dst_path = './'

m = loadmat('nodes_info.mat')
print('type(m) = %s' % type(m))
print(m.keys())
feature_same = m['feature_same']
feature_dif = m['feature_dif']

gcn_features = np.concatenate((feature_same, feature_dif), 0)
real_num = len(gcn_features)

random.seed(100)

print('len(gcn_features) = %s' % len(gcn_features))
print('gcn_features[0].size = %d' % gcn_features[0].size)
savemat(os.path.join(dst_path, 'chen_features.mat'), {'features': gcn_features})


real_gcn_labels = m['train_label']
gen_gcn_labels = m['gen_label']
gcn_labels = np.concatenate((real_gcn_labels, gen_gcn_labels), 1)

print('lable_original = %s' % (gcn_labels))
print('sort_lable = %s' % np.sort(gcn_labels))
print('unique lable = %s' % np.unique(np.sort(gcn_labels[0])))
print('len(gcn_labels) = %s' % len(gcn_labels))
print('len(gcn_labels[0]) = %s' % len(gcn_labels[0]))
print('size = %s' % gcn_labels.size)


label_m = np.zeros(shape=(1, len(gcn_labels[0])), dtype=np.int32)
label_m[0] = gcn_labels[0]
df_index = pd.DataFrame(label_m)
savemat(os.path.join(dst_path, 'chen_label.mat'), {'label': gcn_labels[0]})


# assert 0

def adj_matrix(features):
    global real_num
    num_item = len(features)
    print('num_item = %s' % num_item)
    # assert num_item == 12936
    adj_m = np.zeros(shape=(num_item, num_item), dtype=np.float32)
    adj_row = np.zeros(shape=(5, num_item), dtype=np.float32)
    print('adj_m.ndim =%s ' % adj_m.ndim)
    print('adj_m.size =%s ' % adj_m.size)
    print('adj_m.nbytes =%s ' % adj_m.nbytes)
    time0 = time.time()
    time1 = time0
    preprocess_num = 1

    for i in range(num_item):
    # for i in range(preprocess_num, preprocess_num+1):  # only for test
        time2 = time.time()
        if i % 10 == 0:
            print('%4d epoch  time: %s' % (i, (time2 - time1)))
        time1 = time2
        for j in range(i):
        # for j in range(num_item):  # only for test
            dif_feature = features[i] - features[j]
            temp_adj_value = dif_feature ** 2
            distance = np.sum(temp_adj_value)
            adj_m[i][j] = distance
            adj_m[j][i] = adj_m[i][j]

    # for i in range(num_item):
    #     arr_index = np.argsort(adj_m[i])
    #     adj_m[i][arr_index[100:-1]] = 0 # 101 elements not zero(include itself)
    #     adj_m[i][arr_index[1:100]] = np.divide(1.0, adj_m[i][arr_index[1:100]])
    joint_num = 50
    sparse_num_real = int(joint_num * (float(real_num)/len(gcn_labels[0])))
    sparse_num_gen = joint_num - sparse_num_real
    print('sparse_num_real = %s   sparse_num_gen = %s' % (sparse_num_real, sparse_num_gen))
    # assert sparse_num_real < real_num and sparse_num_gen < len(gcn_labels[0]) - real_num

    for i in range(num_item):
    # for i in range(preprocess_num, preprocess_num+1):  # only for test
        arr_index_real = np.argsort(adj_m[i][: real_num])
        arr_index_gen = np.argsort(adj_m[i][real_num:]) + real_num
        # print('arr_index_real[:sparse_num] = \n%s' % arr_index_real[:sparse_num])
        # print('arr_index_gen[:sparse_num] = \n%s' % arr_index_gen[:sparse_num])
        # print('arr_index_real[sparse_num:] = \n%s' % arr_index_real[sparse_num:])
        # print('arr_index_gen[sparse_num:] = \n%s' % arr_index_gen[sparse_num:])
        adj_m[i][arr_index_real[sparse_num_real:]] = 0  # 101 elements not zero(include itself)
        adj_m[i][arr_index_gen[sparse_num_gen:]] = 0  # 101 elements not zero(include itself)
        adj_m[i][arr_index_real[:sparse_num_real]] = np.exp(-adj_m[i][arr_index_real[:sparse_num_real]])
        adj_m[i][arr_index_gen[:sparse_num_gen]] = np.exp(-adj_m[i][arr_index_gen[:sparse_num_gen]])

    for i in range(num_item):
        for j in range(num_item):
            if math.fabs(adj_m[i][j]) > 1e-09:
                adj_m[j][i] = adj_m[i][j]
            elif math.fabs(adj_m[j][i]) > 1e-09:
                adj_m[i][j] = adj_m[j][i]

    time3 = time.time()
    savemat(os.path.join(dst_path, 'chen_feature_adj.mat'), {'adj': adj_m})

    print('total time = %s' % (time3 - time0))


adj_matrix(gcn_features)

