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
import torch.nn as nn
import torch.nn.functional as F
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
a = []
input = torch.randn(48, 512)
for i in range(128):
    a.append(input[0][i*4: i*4+4].mean())
print(input.shape)
output = F.adaptive_avg_pool1d(input.unsqueeze(0), 128).squeeze(0)
# output = F.avg_pool1d(input.unsqueeze(0), 4)
# print(output.shape)
# output = output.squeeze(0)
print(output.shape)