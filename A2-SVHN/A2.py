# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 20:26:56 2018

@author: user
"""
# For the purpose of viewing image
# import matplotlib.pyplot as plt

import numpy as np
import scipy.io as sio


# Reading data from file
train_mat = sio.loadmat("train_32x32.mat")
test_mat = sio.loadmat("test_32x32.mat")

# Training data, 'raw_data' stands for data with original shape
raw_train_x = train_mat['X']
train_y = train_mat['y']

# Testing data, 'raw_data' stands for data with original shape
raw_test_x = test_mat['X']
test_y = test_mat['y']

