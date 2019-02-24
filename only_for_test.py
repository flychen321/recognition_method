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

# np.random.seed(1)
print(np.random.randint(10))