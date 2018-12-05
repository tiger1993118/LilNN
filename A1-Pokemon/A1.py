# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 18:09:16 2018

NN with Pokemon file
@author: Tiger
"""

import pandas as pd
import numpy as np
import keras.models as km
import keras.layers as kl
import keras.utils as ku
from keras import optimizers

def get_model(numnodes, numnodes2, input_size = 784, output_size = 10):

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
    x1 = kl.Dense(numnodes,
                 activation = 'tanh')(input_image)
    
    # Add 2nd hidden fully-connected layer.
    x = kl.Dense(numnodes2,
                 activation = 'sigmoid')(x1)
    
    # Add the output layer.
    x = kl.Dense(output_size, activation = 'softmax')(x)

    model = km.Model(inputs = input_image,
                     outputs = x)
    
    # Return the model.
    return model

"""Read data, then reshape and divide into test and train"""
# Read data
data = pd.read_csv('pokemon.csv').values;
x = data[:, 1:19].astype('float32')
y = data[:, 39].astype('uint8')-1# -1 because to_categorical starting at 0
x = x * 4 #integer

# Size and dimensions
size = y.shape[0] #801
input_dim = x.shape[1]
output_dim = 7

# Non-repeated randomly select 40 indices as test set
test_choice = np.random.choice(size, 60, replace = False)

# Test data
test_x = np.take(x, test_choice, axis = 0)
test_y = np.take(y, test_choice)

# Train data
train_x = np.delete(x, test_choice, axis = 0) # Train = Total - Test
train_y = np.delete(y, test_choice)

# Reshape y to 1 of k coding
test_y = ku.to_categorical(test_y, output_dim)
train_y = ku.to_categorical(train_y, output_dim)

"""Predict using model2"""

model = get_model(35, 20, input_dim, output_dim)

# Learning rate, decay
sgd = optimizers.SGD(lr=0.01, decay=1e-4)
model.compile(optimizer=sgd, metrics = ['accuracy'], loss = "mean_squared_error")
fit = model.fit(train_x, train_y, epochs=800, batch_size=5, verbose=2)

sgd = optimizers.SGD(lr=0.007, decay=1e-3)
model.compile(optimizer=sgd, metrics = ['accuracy'], loss = "mean_squared_error")
fit = model.fit(train_x, train_y, epochs=400, batch_size=5, verbose=2)

score = model.evaluate(test_x, test_y)



































