# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 18:09:16 2018

NN with Pokemon file
@author: Tiger
"""

from timeit import default_timer as timer
import pandas as pd
import numpy as np
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

# Read data
data = pd.read_csv('pokemon.csv').values;
x = data[:, 1:19].astype('float32')
y = data[:, 39].astype('uint8')-1# -1 because to_categorical starting at 0
x = x * 4 #integer

# Size and dimensions
size = y.shape[0] #801
input_dim = x.shape[1]
output_dim = 7
    
# Storing parameters
train = []
test = []
lrs = []
decays = []
numnode = []

lr = 0.0104
decay = 1e-6
nodes2 = 27

for i in range(200):
    
#    elif(i <= 500):
#        lr = 0.01
#        if(i % 5 == 0):
#            decay += 2e-7   
    
    if(i % 5 == 0):
        nodes2 -= 1
        
    # Set up Timer
    start = timer()
    """Read data, then reshape and divide into test and train"""
    
    # Non-repeated randomly select 40 indices as test set
    test_choice = np.random.choice(size, 100, replace = False)
    
    # Test data
    test_x = np.take(x, test_choice, axis = 0)
    test_y = np.take(y, test_choice)
    
    # Train data
    train_x = np.delete(x, test_choice, axis = 0) # Train = Total - Test
    train_y = np.delete(y, test_choice)
    
    # Print number of occurances of each category
    unique, counts = np.unique(test_y, return_counts=True)
    count_map = dict(zip(unique, counts))
    #print(count_map)
    
    # Reshape y to 1 of k coding
    test_y = ku.to_categorical(test_y, output_dim)
    train_y = ku.to_categorical(train_y, output_dim)
    
    """Predict using model2"""
    
    model = get_model(50, input_dim, output_dim, nodes2)
    
    # Learning rate, decay
    sgd = optimizers.SGD(lr=lr, decay=decay)
    model.compile(optimizer=sgd, metrics = ['accuracy'], loss = "mean_squared_error")
    model.fit(train_x, train_y, epochs=2000, batch_size=5, verbose=0)
    
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
    
    lrs.append(lr)
    decays.append(decay)
    numnode.append(nodes2)
    train.append(train_score)
    test.append(score)
    end = timer()
    
    print("i", i, "time", round(end - start, 0), "lr", round(lr, 9) \
          , "decay", decay, "nodes2", nodes2 \
          , "train", round(train_score, 3), "test", round(score, 2))
    
    
'''       
After Testing

From i = 0 to 250 
which lr = 0.0081 to 0.013
lrs[102:127] having the best range of correctness
[0.35, 0.34, 0.32, 0.29, 0.32, 0.29, 0.32, 0.29, 0.32, 0.35, 0.33,
 0.36, 0.29, 0.34, 0.3 , 0.26, 0.3 , 0.3 , 0.35, 0.35, 0.29, 0.27,
 0.31, 0.28, 0.29]
for which lr = 0.0102 to 0.0106
I will choose lr = 0.0104

From i = 0 to 200
which numnode = 40 to 1
numnodes[63:79] have the best range of correctnes
[0.33, 0.31, 0.3 , 0.37, 0.3 , 0.31, 0.3 , 0.27, 0.34, 0.28, 0.29,
0.28, 0.34, 0.35, 0.35, 0.3 ]
I will choose numnodes = 27
'''


































