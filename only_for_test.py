import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable
import os
import numpy as np
import math
import scipy.sparse as sp
import torch.nn.functional as F


# a = torch.Tensor([[1,2,3], [1,2,8]])
# print(a)
# print(F.softmax(a, 0))
# print(F.softmax(a, 0)[:,0].sum())

# index = np.arange(48)
# np.random.shuffle(index[16:])
# print(index)
# print(np.random.permutation(10))

num_p_per_batch = 128
adj = torch.FloatTensor(num_p_per_batch, num_p_per_batch).cuda().fill_(
            1.0 / (num_p_per_batch * num_p_per_batch))

print(np.random.randint(num_p_per_batch*0.2, num_p_per_batch*0.6))