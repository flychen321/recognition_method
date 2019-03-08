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

src_path = 'data/filter_data/train_all_2027'
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
    return real_num, fake_num, file_num


def sample(path='data/filter_data', num=5000):
    files = os.listdir(os.path.join(path, 'real'))
    # files = np.sort(files)
    np.random.shuffle(files)
    dst_sub_dir = 'real_sample'
    if not os.path.exists(os.path.join('data/filter_data', dst_sub_dir)):
        os.makedirs(os.path.join('data/filter_data', dst_sub_dir))
    for i in range(num):
        shutil.copy(os.path.join(os.path.join(path, 'real'), files[i]), os.path.join('data/filter_data', dst_sub_dir, files[i]))

    files = os.listdir(os.path.join(path, 'fake'))
    # files = np.sort(files)
    np.random.shuffle(files)
    dst_sub_dir = 'fake_sample'
    if not os.path.exists(os.path.join('data/filter_data', dst_sub_dir)):
        os.makedirs(os.path.join('data/filter_data', dst_sub_dir))
    for i in range(num):
        shutil.copy(os.path.join(os.path.join(path, 'fake'), files[i]), os.path.join('data/filter_data', dst_sub_dir, files[i]))


def redivide(path='data/filter_data', ratio=0.9):
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

    files = os.listdir(os.path.join(path, 'real_sample'))
    # files = np.sort(files)
    np.random.shuffle(files)
    for i in range(len(files)):
        if i < int(ratio * len(files)):
            shutil.copy(os.path.join(path, 'real_sample', files[i]), os.path.join(dst_base_path1, dst_sub_dir1, files[i]))
        else:
            shutil.copy(os.path.join(path, 'real_sample', files[i]), os.path.join(dst_base_path2, dst_sub_dir1, files[i]))

    files = os.listdir(os.path.join(path, 'fake_sample'))
    # files = np.sort(files)
    np.random.shuffle(files)
    for i in range(len(files)):
        if i < int(ratio * len(files)):
            shutil.copy(os.path.join(path, 'fake_sample', files[i]), os.path.join(dst_base_path1, dst_sub_dir2, files[i]))
        else:
            shutil.copy(os.path.join(path, 'fake_sample', files[i]), os.path.join(dst_base_path2, dst_sub_dir2, files[i]))

def for_train():
    real_num, fake_num, file_num = generate_dataset()
    sample(path='data/filter_data', num=min(real_num, fake_num))
    redivide()

def for_test():
    dirs = os.listdir(src_path)
    dst_path = os.path.join(dst_base_path, 'test_set/fake')
    print(len(dirs))
    if os.path.exists(dst_path):
        shutil.rmtree(dst_path)
    os.makedirs(os.path.join(dst_path))
    file_num = 0
    dir_num = 0
    for dir in dirs:
        if len(dir) == 8:
            files = os.listdir(os.path.join(src_path, dir))
            file_num += len(files)
            dir_num += 1
            for file in files:
                shutil.copy(os.path.join(src_path, dir, file), os.path.join(dst_path, file))
    print('dir_num = %d    file_num = %d' % (dir_num, file_num))


def for_reid():
    sample_good = os.path.join(dst_base_path, 'good')
    dst_path = os.path.join(dst_base_path, 'augment')
    if not os.path.exists(sample_good):
        print('src data %s is not exist' % sample_good)
    if os.path.exists(dst_path):
        shutil.rmtree(dst_path)
    os.mkdir(dst_path)
    files = os.listdir(sample_good)
    file_num = 0
    dir_num = 0
    for file in files:
        dir = file.split('_')[0]
        if len(dir) == 8:
            if not os.path.exists(os.path.join(dst_path, dir)):
                os.mkdir(os.path.join(dst_path, dir))
                dir_num += 1
            shutil.copy(os.path.join(sample_good, file), os.path.join(dst_path, dir, file))
            file_num += 1
    print('after first filter total file_num = %d  dir_num = %d' % (file_num, dir_num))

    dirs = os.listdir(dst_path)
    for dir in dirs:
        files = os.listdir(os.path.join(dst_path, dir))
        if len(files) < 4:
            file_num -= len(files)
            dir_num -= 1
            shutil.rmtree(os.path.join(dst_path, dir))

    print('after second filter total file_num = %d  dir_num = %d' % (file_num, dir_num))





if __name__ == '__main__':
    # for_train()
    # for_test()
    for_reid()



