import xlrd
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.io import savemat
import time
import random
import os
import h5py
import shutil

src_path_train_index = 'data/MSMT17_V2/list_train.txt'
src_path_val_index = 'data/MSMT17_V2/list_val.txt'
src_path_gallery_index = 'data/MSMT17_V2/list_gallery.txt'
src_path_query_index = 'data/MSMT17_V2/list_query.txt'
src_path_train_data = 'data/MSMT17_V2/mask_train_v2'
src_path_test_data = 'data/MSMT17_V2/mask_test_v2'
dst_path_train = 'data/MSMT17_V2/bounding_box_train'
dst_path_test = 'data/MSMT17_V2/bounding_box_test'
dst_path_query = 'data/MSMT17_V2/query'

if os.path.exists(dst_path_train):
    shutil.rmtree(dst_path_train)
os.mkdir(dst_path_train)
if os.path.exists(dst_path_test):
    shutil.rmtree(dst_path_test)
os.mkdir(dst_path_test)
if os.path.exists(dst_path_query):
    shutil.rmtree(dst_path_query)
os.mkdir(dst_path_query)

train_cnt = 0
gallery_cnt = 0
query_cnt = 0
with open(src_path_train_index) as f:
    while True:
        line = f.readline()
        if not line:
            break
        line = line[:-1]
        # print('src = %s' % os.path.join(src_path_data, line.split(' ')[0]))
        # print('dst = %s' % os.path.join(dst_path_train, os.path.split(line.split(' ')[0])[-1]))
        shutil.copy(os.path.join(src_path_train_data, line.split(' ')[0]), os.path.join(dst_path_train, os.path.split(line.split(' ')[0])[-1]))
        train_cnt += 1
    print('train_cnt = %d' % train_cnt)

with open(src_path_val_index) as f:
    while True:
        line = f.readline()
        if not line:
            break
        line = line[:-1]
        # print('src = %s' % os.path.join(src_path_data, line.split(' ')[0]))
        # print('dst = %s' % os.path.join(dst_path_train, os.path.split(line.split(' ')[0])[-1]))
        shutil.copy(os.path.join(src_path_train_data, line.split(' ')[0]), os.path.join(dst_path_train, os.path.split(line.split(' ')[0])[-1]))
        train_cnt += 1
    print('train_cnt = %d' % train_cnt)

with open(src_path_gallery_index) as f:
    while True:
        line = f.readline()
        if not line:
            break
        line = line[:-1]
        # print('src = %s' % os.path.join(src_path_data, line.split(' ')[0]))
        # print('dst = %s' % os.path.join(dst_path_train, os.path.split(line.split(' ')[0])[-1]))
        shutil.copy(os.path.join(src_path_test_data, line.split(' ')[0]), os.path.join(dst_path_test, os.path.split(line.split(' ')[0])[-1]))
        gallery_cnt += 1
    print('gallery_cnt = %d' % gallery_cnt)

with open(src_path_query_index) as f:
    while True:
        line = f.readline()
        if not line:
            break
        line = line[:-1]
        # print('src = %s' % os.path.join(src_path_data, line.split(' ')[0]))
        # print('dst = %s' % os.path.join(dst_path_train, os.path.split(line.split(' ')[0])[-1]))
        shutil.copy(os.path.join(src_path_test_data, line.split(' ')[0]), os.path.join(dst_path_query, os.path.split(line.split(' ')[0])[-1]))
        query_cnt += 1
    print('query_cnt = %d' % query_cnt)