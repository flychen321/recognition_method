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

path = 'data/market/pytorch/shuffle_test'
# files = os.listdir(path)
# for file in files:
#     img = cv2.imread(os.path.join(path, file))
#     cv2.imshow('org', img)
#     cv2.waitKey(1000)
#     img = np.array(img)
#     # np.random.shuffle(img)
#     # img = np.transpose(img, (1, 0, 2))
#     # np.random.shuffle(img)
#     # img = np.transpose(img, (1, 0, 2))
#     # index = np.random.permutation(3)
#     # print(index)
#     # print((index == np.arange(3)).all())
#     # img = np.transpose(img, (2, 1, 0))
#     # np.random.shuffle(img)
#     # img = np.transpose(img, (2, 1, 0))
#     img = (img*0.5).astype(np.uint8)
#     b = img
#     # b = np.concatenate((img[:, int(img.shape[1]/2):, :], img[:, :int(img.shape[1]/2), :]), 1)
#     print(b.shape)
#     cv2.imshow('new', b)
#     cv2.waitKey(5000)

a = np.random.permutation(10)
print(a)
print('0002_c1s1_000451_03.jpg')
print('0002_c1s1_000451_03.jpg'.split('c'))