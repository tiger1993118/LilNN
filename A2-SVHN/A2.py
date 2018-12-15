"""
The NN model functon for A2
"""

import keras.models as km
import keras.layers as kl


def get_model(num_fm, num_nodes, input_shape=(32, 32, 3, 1), output_size=10):
    
    # Initialize the model
    model = km.Sequential()
    
    # Add a 2D convolution layer, with num_fm feature maps
    model.add(kl.Conv3D(num_fm, 
                        kernel_size=(5, 5, 3),
                        input_shape=input_shape,
                        activation = 'relu'))
    # Output is (28, 28, 3, num_fm)
    
    
    # Add a max pooling layer
    model.add(kl.MaxPool3D(pool_size=(2, 2, 1),
                           strides=(2, 2, 1)))
    # Output is (14, 14, 3, num_fm)
        
        
    # Add a 2D convolution layer, with num_fm feature maps
    model.add(kl.Conv3D(num_fm*2, 
                        kernel_size=(5, 5, 1),
                        input_shape=input_shape,
                        activation = 'relu'))
    # Output is (10, 10, 3, num_fm)
    
    
    # Add a max pooling layer
    model.add(kl.MaxPool3D(pool_size=(2, 2, 1),
                           strides=(2, 2, 1)))
    # Output is (5, 5, 3, num_fm)
    

    model.add(kl.Flatten())
    
    # Add a fully-connected layer.
    model.add(kl.Dense(num_nodes, activation = 'tanh'))

    # Add the output layer.
    model.add(kl.Dense(10, activation = 'softmax'))

    # Return the model.
    return model



def get_model2(numfm, numnodes, input_shape = (32, 32, 1),
              output_size = 10):

    """
    This function returns a convolution neural network Keras model,
    with numfm feature maps in the first convolutional layer, 2 *
    numfm in the second convolutional layer, and numnodes neurons in
    the fully-connected layer.

    Inputs:
    - numfm: int, the number of feature maps in the convolution layer.

    - numnodes: int, the number of nodes in the fully-connected layer.

    - intput_shape: tuple, the shape of the input data, 
    default = (28, 28, 1).

    - output_size: int, the number of nodes in the output layer,
      default = 10.

    Output: the constructed Keras model.

    """

    # Initialize the model.
    model = km.Sequential()
    
    # Input is 28x28
    # Add a 2D convolution layer, with numfm feature maps.
    model.add(kl.Conv2D(numfm, kernel_size = (5, 5),
                        input_shape = input_shape,
                        activation = 'relu'))
    
    # Input is 24x24
    # Add a max pooling layer.
    model.add(kl.MaxPooling2D(pool_size = (2, 2),
                              strides = (2, 2)))
    
    # Input is 12x12
    # Add a 2D convolution layer, with 2xnumfm feature maps.
    model.add(kl.Conv2D(numfm * 2, kernel_size = (3, 3),
                        activation = 'relu'))

    # Input is 10x10
    # Add a max pooling layer.
    model.add(kl.MaxPooling2D(pool_size = (2, 2),
                              strides = (2, 2)))
    
    # Input is 5x5
    # Convert the network from 2D to 1D.
    model.add(kl.Flatten())
    
    # Add a fully-connected layer.
    model.add(kl.Dense(numnodes, activation = 'tanh'))

    # Add the output layer.
    model.add(kl.Dense(10, activation = 'softmax'))

    # Return the model.
    return model