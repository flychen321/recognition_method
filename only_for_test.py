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
import shutil
import math
import matplotlib.pyplot as plt
#
# a = np.arange(0.001, 1, 0.01)
# # print(a)
# b = [-x*np.log(x) for x in a]
# c = [-np.log(x) for x in a]
# print(b)
# print(len(b))
# print(np.argmax(b))
# print(np.max(b))
# plt.subplot(311)
# plt.plot(a, b, 'r')
# plt.subplot(312)
# plt.plot(a, c, 'g')
# plt.subplot(313)
# plt.plot(a, a, 'b')
# plt.show()

# p1 = 0.9
# p2 = (1-p1)
#
# for i in np.arange(0.1, 1, 0.01):
#     l1 = i
#     l2 = (1-l1)
#     crossentropy = -(l1*np.log(p1) + l2*np.log(p2))
#     print('l1 = %.5f  crossentropy = %.5f' % (l1, crossentropy))


# m = nn.Sigmoid()
loss = nn.BCELoss(size_average=False, reduce=False)
# input = torch.randn(3)
# target = torch.empty(3).random_(2)
# lossinput = m(input)
# output = loss(lossinput, target)
# print(input)
# print(lossinput)
# print(target)
# print(output)
# print(-(target[0]*math.log(lossinput[0]) + (1-target[0])*math.log(1-lossinput[0])))
input = torch.randn(4, 7)
label = torch.empty(4, 7).random_(2)
weight = torch.rand(4, 7)
print(input)
print(label)
print(weight)

def loss_bce(input, label, weight, reduce=True):
    result = F.binary_cross_entropy_with_logits(input, label, weight, reduce=False)
    result2 = F.binary_cross_entropy_with_logits(input, label, reduce=False)
    result3 = F.binary_cross_entropy_with_logits(input, label, weight, reduce=True)
    result4 = F.binary_cross_entropy_with_logits(input, label, weight, reduction='elementwise_mean')
    result5 = F.binary_cross_entropy_with_logits(input, label, weight, reduction='none')
    result6 = F.binary_cross_entropy_with_logits(input, label, weight, reduction='sum')
    print(result)
    print(result2)
    print(result3)
    print(result4)
    print(result5)
    print(result6)
    input = torch.sigmoid(input)
    print('my owner manner')
    print(-weight[0, 0] * (label[0, 0] * math.log(input[0, 0]) + (1 - label[0, 0]) * math.log(1 - input[0, 0])))
    return result

if __name__ == '__main__':
    r = loss_bce(input, label, weight)









































