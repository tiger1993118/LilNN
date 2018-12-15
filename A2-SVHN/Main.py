
"""
Process: 

1. Applying CNN to the data and check the runtime and result

"""

import scipy.io as sio
import numpy as np
import keras.utils as ku
from A2 import get_model
from timeit import default_timer as timer
#from matplotlib.pyplot import imshow


#################### Read Data and Processing ################################

# Reading data from local file
train_mat = sio.loadmat("train_32x32.mat")
test_mat = sio.loadmat("test_32x32.mat")

# X values -- Training data, 'raw_data' stands for data with original shape
raw_train_x = train_mat['X']# (32, 32, 3, 73257)
raw_test_x = test_mat['X']# (32, 32, 3, 26032)
# Reshaping the X to roll the last value to the 1st 
train_x = np.rollaxis(raw_train_x, 3)
test_x = np.rollaxis(raw_test_x, 3)
# Reshaping the X to add 1 dimension at the end 
train_x = np.expand_dims(train_x, axis = 4)
test_x = np.expand_dims(test_x, axis = 4)
#train_x = train_x[:, :, :, 0:1]
#test_x = test_x[:, :, :, 0:1]

# Y values -- Testing data, 'raw_data' stands for data with original shape
train_y = train_mat['y']# (73257, 1)
test_y = test_mat['y']# (26032, 1)
# Y values -- Reshape Y values to 1-of-K coding
#Change 10 to 0, which represents digit 0
train_y[train_y == 10] = 0 
train_y = ku.to_categorical(train_y, 10)# (73257, 10)
test_y[test_y == 10] = 0
test_y = ku.to_categorical(test_y, 10)# (73257, 10)

######################### Set up the NN model ################################

model = get_model(20, 100)

# Compile the NN model 
model.compile(loss = "categorical_crossentropy", 
              optimizer = "adam", 
              metrics = ['accuracy'])


######################### Training ##########################################

# Start Timer
start = timer()

# Training Process
fit = model.fit(train_x, train_y, epochs=50, batch_size=5000, verbose=2)

# End Time
end = timer()

# Test Accuracy
score = model.evaluate(test_x, test_y)


print("Time is: ", round((end - start), 2), " seconds")


