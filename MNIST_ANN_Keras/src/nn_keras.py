
# Implementing Neural Nets using Keras on MNIST dataset
# Author: Karthikeya Bolla

# Tutorial to implement using Keras http://machinelearningmastery.com/tutorial-first-neural-network-python-keras/

import keras
import idx2numpy
import numpy as np
from keras import optimizers
from keras import regularizers
from keras.models import Sequential 		# sequential stack of layers
from keras.layers import Dense			# layers that are completely connected
from keras.utils import np_utils
from keras.layers import Activation
from sklearn.model_selection import KFold

# =======================================================================
# 				training dataset
# =======================================================================
images = idx2numpy.convert_from_file('/home/karthikeya/Desktop/nn/train-images.idx3-ubyte')
labels = idx2numpy.convert_from_file('/home/karthikeya/Desktop/nn/train-labels.idx1-ubyte')

train_data = []
for i in range(len(images)):
	train_data.append(images[i].flatten())

train_data = np.asarray(train_data)
labels = labels.reshape(train_data.shape[0], 1)			# reshaping y_train from (60000, ) to (60000, 1)
train_data = np.concatenate((train_data, labels), axis = 1)	# creating 60000 X 785 matrix with last column being class labels

np.random.seed(1)						# random seed
np.random.shuffle(train_data)					# shuffle dataset

# =======================================================================
# 				Architecture
# =======================================================================
# This is a 2-layer NN. Input layer has 784 neurons as the dataset dimensions in 784
# Hidden layer has 1000 neurons and output layer has 10 neurons as the output is digits 0 - 9
# ReLU activation used in hidden layer
# Softmax function used in output layer

model = Sequential()

# adding hidden layer to model
model.add(Dense(1000, 
		input_dim = 784, 
		activation = 'relu', 
		use_bias = True, 
		bias_initializer = 'zeros',
		kernel_initializer = 'he_normal', 
		kernel_regularizer = regularizers.l2(0.001)))		

# adding output layer to model
model.add(Dense(10, 
		activation = 'softmax',
		use_bias = True,
		bias_initializer = 'zeros',
		kernel_initializer = 'glorot_normal',
		kernel_regularizer = regularizers.l2(0.001)))

# =======================================================================
# 			Compile the model
# =======================================================================
sgd = optimizers.SGD(lr = 0.09, clipnorm = 1.0)							# set the learning rate and use SGD
model.compile(loss = 'categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])	# compile the model

# =======================================================================
# 		Run the model on training dataset
# =======================================================================
y = train_data[:, -1]				# select last column which is column containing labels (60000 X 1)
y_categ = keras.utils.to_categorical(y)		# convert class values to one-hot encoding i.e. convert to categorical data (60000 X 10) 
X = np.delete(train_data, -1, axis = 1)		# remove last column and this becomes training input (60000 X 784)
X = X/255.0

kf = KFold(n_splits = 10)					# 10-fold cross-validation
for train_index, valid_index in kf.split(train_data):

	train = train_data[train_index]				# 54000 X 785 matrix of training data
	valid = train_data[valid_index]				# 6000 X 785 matrix of validation data

	y_train = train[:, -1]					# select last column containing labels (54000 X 1)
	y_train_categ = keras.utils.to_categorical(y_train)	# convert class vals to categorical data (54000 X 10)	
	X_train = np.delete(train, -1, axis = 1)		# remove last column and this becomes training input (54000 X 784)
	X_train = X_train/255.0

	y_valid = valid[:, -1]					# select last column which is column containing labels (6000 X 1)
	y_valid_categ = keras.utils.to_categorical(y_valid)	# convert class vals to categorical data (6000 X 10)	
	X_valid = np.delete(valid, -1, axis = 1)		# remove last column and this becomes training input (6000 X 784)
	X_valid = X_valid/255.0

	# fit the model
	model.fit(X_train, y_train_categ, batch_size = 30, epochs = 1, validation_data = (X_valid, y_valid_categ))

# =======================================================================
# 			Run the model on test dataset
# =======================================================================
test_images = idx2numpy.convert_from_file('/home/karthikeya/Desktop/nn/t10k-images.idx3-ubyte')
test_labels = idx2numpy.convert_from_file('/home/karthikeya/Desktop/nn/t10k-labels.idx1-ubyte')

# converting test dataset from (10000 X 28 X 28) to (10000 X 784)
test = []
for i in range(len(test_images)):
	test.append(test_images[i].flatten())

test = np.asarray(test)						# test dataset of dimensiosn (10000 X 784)
test = test/255.0	

predict = model.predict(test)					# this is of size 10000 X 784
predict = np.argmax(predict, axis = 1)				# fetching index of maximum element in every row	
mean = np.mean(predict == test_labels)
print "test accuracy: ", mean



