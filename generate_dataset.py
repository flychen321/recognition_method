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

src_path = 'data/filter_data/train_all_1072'
dst_base_path = 'data/filter_data/'

def generate_dataset():
    dirs = os.listdir(src_path)
    print(len(dirs))
    dst_sub_dir1 = 'real'
    dst_sub_dir2 = 'fake'
    if not os.path.exists(os.path.join(dst_base_path, dst_sub_dir1)):
        os.makedirs(os.path.join(dst_base_path, dst_sub_dir1))
    if not os.path.exists(os.path.join(dst_base_path, dst_sub_dir2)):
        os.makedirs(os.path.join(dst_base_path, dst_sub_dir2))
    file_num = 0
    real_num = 0
    fake_num = 0
    for dir in dirs:
        files = os.listdir(os.path.join(src_path, dir))
        # print(len(files))
        if len(dir) == 4:
            real_num += len(files)
            for file in files:
                shutil.copy(os.path.join(src_path, dir, file), os.path.join(dst_base_path, dst_sub_dir1, file))
        elif len(dir) == 8:
            fake_num += len(files)
            for file in files:
                shutil.copy(os.path.join(src_path, dir, file), os.path.join(dst_base_path, dst_sub_dir2, file))
        file_num += len(files)
    print('real = %4d  fake = %4d  sum = %4d' % (real_num, fake_num, file_num))


def sample(path = 'data/filter_data/real_sum', num = 5000):
    files = os.listdir(path)
    np.random.shuffle(files)
    dst_sub_dir = 'real'
    if not os.path.exists(os.path.join('data/filter_data', dst_sub_dir)):
        os.makedirs(os.path.join('data/filter_data', dst_sub_dir))
    for i in range(num):
        shutil.copy(os.path.join(path, files[i]), os.path.join('data/filter_data', dst_sub_dir, files[i]))


def redivide(path = 'data/filter_data', ratio = 0.8):
    dst_base_path1 = os.path.join(path, 'train_set')
    dst_base_path2 = os.path.join(path, 'test_set')
    dst_sub_dir1 = 'real'
    dst_sub_dir2 = 'fake'
    if not os.path.exists(dst_base_path1):
        os.makedirs(dst_base_path1)
    if not os.path.exists(os.path.join(dst_base_path1, dst_sub_dir1)):
        os.makedirs(os.path.join(dst_base_path1, dst_sub_dir1))
    if not os.path.exists(os.path.join(dst_base_path1, dst_sub_dir2)):
        os.makedirs(os.path.join(dst_base_path1, dst_sub_dir2))
    if not os.path.exists(dst_base_path2):
        os.makedirs(dst_base_path2)
    if not os.path.exists(os.path.join(dst_base_path2, dst_sub_dir1)):
        os.makedirs(os.path.join(dst_base_path2, dst_sub_dir1))
    if not os.path.exists(os.path.join(dst_base_path2, dst_sub_dir2)):
        os.makedirs(os.path.join(dst_base_path2, dst_sub_dir2))

    files = os.listdir(os.path.join(path, 'real'))
    np.random.shuffle(files)
    for i in range(len(files)):
        if i < int(ratio * len(files)):
            shutil.copy(os.path.join(path, 'real', files[i]), os.path.join(dst_base_path1, dst_sub_dir1, files[i]))
        else:
            shutil.copy(os.path.join(path, 'real', files[i]), os.path.join(dst_base_path2, dst_sub_dir1, files[i]))

    files = os.listdir(os.path.join(path, 'fake'))
    np.random.shuffle(files)
    for i in range(len(files)):
        if i < int(ratio * len(files)):
            shutil.copy(os.path.join(path, 'fake', files[i]), os.path.join(dst_base_path1, dst_sub_dir2, files[i]))
        else:
            shutil.copy(os.path.join(path, 'fake', files[i]), os.path.join(dst_base_path2, dst_sub_dir2, files[i]))



if __name__ == '__main__':
    # sample()
    redivide()



