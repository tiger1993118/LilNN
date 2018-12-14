# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 20:26:56 2018

@author: user
"""
# For the purpose of viewing image
# import matplotlib.pyplot as plt
# plt.imshow(image)


import numpy as np
import scipy.io as sio


from timeit import default_timer as timer
import pandas as pd
import keras.models as km
import keras.layers as kl
import keras.utils as ku
from keras import optimizers
from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications.vgg16 import VGG16

# Reading data from file
train_mat = sio.loadmat("train_32x32.mat")
test_mat = sio.loadmat("test_32x32.mat")

# Training data, 'raw_data' stands for data with original shape
raw_train_x = train_mat['X']
train_y = train_mat['y']

# Testing data, 'raw_data' stands for data with original shape
raw_test_x = test_mat['X']
test_y = test_mat['y']


'''##########################################################'''
input_dim = 32*32*3
output_dim = 10

# Reshape x
train_x = np.rollaxis(raw_train_x, 3, 0)
#train_x = train_x.reshape(train_x.shape[0], -1)
test_x = np.rollaxis(raw_test_x, 3, 0)
#test_x = test_x.reshape(test_x.shape[0], -1)


# Reshape y to 1 of k coding
test_y = ku.to_categorical(test_y-1, output_dim)
train_y = ku.to_categorical(train_y-1, output_dim)
'''##########################################################'''

base = VGG16(weights=None, input_shape = (32, 32, 3), \
                 include_top=False)

y = Dense(output_dim, activation="softmax")(base.output)

model = km.Model(base.input, y)

sgd = optimizers.SGD(lr=0.1)
model.compile(optimizer=sgd, metrics = ['accuracy'], loss = "mean_squared_error")
model.fit(train_x, train_y, epochs=1000, batch_size=5, verbose=2)

#model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

#model.fit_generator(...)











