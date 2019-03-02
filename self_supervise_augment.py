import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.io import savemat
import time
import random
import os
import math
import cv2
import torch
# image size: 128 * 64 *3
path = 'data/market/pytorch/train_all_original'
dst_path = 'data/market/bounding_box_train'
# path = 'data/market/pytorch/part'
dirs = os.listdir(path)
dir_len = len(dirs)
new_dir_num = 0
new_file_num = 0
base = 2000
for i in range(int(len(dirs) / 2)):
    print(dirs[i])
    print(dirs[len(dirs) - 1 - i])
    files1 = os.listdir(os.path.join(path, dirs[i]))
    files2 = os.listdir(os.path.join(path, dirs[len(dirs) - 1 - i]))
    num = min(len(files1), len(files2))
    if num < 6:
        continue
    new_dir_num += 1
    new_file_num += num
    index1 = np.random.permutation(len(files1))[:num]
    index2 = np.random.permutation(len(files2))[:num]
    dir_path = os.path.join(path, dirs[i] + dirs[len(dirs) - 1 - i])
    if not os.path.exists(dir_path):
        # os.makedirs(dir_path)
        for j in range(num):
            img1 = cv2.imread(os.path.join(path, dirs[i], files1[index1[j]]))
            img2 = cv2.imread(os.path.join(path, dirs[len(dirs) - 1 - i], files2[index2[j]]))
            img_new = np.concatenate((img1[:int(img1.shape[0] / 2), :, ], img2[int(img2.shape[0] / 2):, :, :]), 0)
            file_name = dirs[i] + dirs[len(dirs) - 1 - i] + '_' + str(j) + '_c' + files1[j].split('c')[1]
            cv2.imwrite(os.path.join(dst_path, file_name), img_new)
            # print(file_name)
            # exit()
            # cv2.imshow('org1', img1)
            # cv2.imshow('org2', img2)
            # cv2.imshow('new', img_new)
            # cv2.waitKey(1000)
print('new_dir_num = %d  new_file_num = %d' % (new_dir_num, new_file_num))
print(len(dirs))
