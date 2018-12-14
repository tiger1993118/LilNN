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


def get_model(numnodes, input_size = 784, output_size = 10, numnodes2=0):

    """
    This function returns a simple Keras model, consisting of a
    re-implementation of the second_network.py neural network, with
    numnodes in the hidden layer, using Keras' functional syntax.

    Inputs:
    - numnodes: int, the number of nodes in the hidden layer.

    - intput_size: int, the size of the input data, default = 784.

    - output_size: int, the number of nodes in the output layer,
      default = 10.

    Output: the constructed Keras model.

    """

    input_image = kl.Input(shape = (input_size,))

    # Add a hidden fully-connected layer.
    x = kl.Dense(numnodes, activation = 'sigmoid')(input_image)
    
    
    # Add 2nd hidden fully-connected layer.
    if(numnodes2 != 0):
        x = kl.Dense(numnodes2, activation = 'tanh')(x)
    
    # Add the output layer.
    x = kl.Dense(output_size, activation = 'softmax')(x)

    model = km.Model(inputs = input_image,
                     outputs = x)
    
    # Return the model.
    return model

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
train_x = train_x.reshape(train_x.shape[0], -1)
test_x = np.rollaxis(raw_test_x, 3, 0)
test_x = test_x.reshape(test_x.shape[0], -1)

# Reshape y to 1 of k coding
test_y = ku.to_categorical(test_y-1, output_dim)
train_y = ku.to_categorical(train_y-1, output_dim)

# Set up Timer
start = timer()

"""Predict using model"""

model = get_model(50, input_dim, output_dim, 30)

# Learning rate, decay
sgd = optimizers.SGD(lr=0.0001)
model.compile(optimizer=sgd, metrics = ['accuracy'], loss = "mean_squared_error")
model.fit(train_x, train_y, epochs=1000, batch_size=20, verbose=2)

'''
sgd = optimizers.SGD(lr=0.005, decay=1e-4)
model.compile(optimizer=sgd, metrics = ['accuracy'], loss = "mean_squared_error")
fit = model.fit(train_x, train_y, epochs=400, batch_size=10, verbose=0)

sgd = optimizers.SGD(lr=0.01, decay=1e-5)
model.compile(optimizer=sgd, metrics = ['accuracy'], loss = "mean_squared_error")
fit = model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=0)
'''
train_score = model.evaluate(train_x, train_y, verbose=0)[1]
score = model.evaluate(test_x, test_y, verbose=0)[1]

end = timer()

print("i", i, "time", round(end - start, 0), "lr", round(lr, 9) \
      , "decay", decay, "nodes2", nodes2 \
      , "train", round(train_score, 3), "test", round(score, 2))
















