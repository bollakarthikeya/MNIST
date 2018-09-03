
# implementing Neural Networks from scratch
# http://cs.stanford.edu/people/karpathy/cs231nfiles/minimal_net.html
# http://briandolhansky.com/blog/2014/10/30/artificial-neural-networks-matrix-form-part-5

import time
import idx2numpy
import numpy as np
from scipy.special import expit
import matplotlib.pyplot as plt

# =======================================================================
# 				training dataset
# =======================================================================
images = idx2numpy.convert_from_file('/home/karthikeya/Desktop/nn/train-images.idx3-ubyte')
labels = idx2numpy.convert_from_file('/home/karthikeya/Desktop/nn/train-labels.idx1-ubyte')

start_time = time.clock()

train_data = []
for i in range(len(images)):
	train_data.append(images[i].flatten())

train_data = np.asarray(train_data)
labels = labels.reshape(train_data.shape[0], 1)			# reshaping y_train from (60000, ) to (60000, 1)
train_data = np.concatenate((train_data, labels), axis = 1)	# creating 60000 X 785 matrix with last column being class labels

# We will use ReLU activation function for the hidden layer
# derivative of ReLU function is http://kawahara.ca/what-is-the-derivative-of-relu/
# We will use Softmax loss function to compute loss
# derivative of softmax loss function is http://cs231n.github.io/neural-networks-case-study/#loss

# =======================================================================
#			initialize parameters
# =======================================================================
h = 1000								# no. of neurons in hidden layer
iterations = 10								# epochs i.e. no. of times dataset is seen by model
D = train_data.shape[1] - 1						# dataset dimension (no. of features)
K = 10									# no. of output neurons (10 for this dataset)
# num_records = X_train.shape[0]						# no. of records in dataset
W_inp_1 = np.random.randn(D, h) * np.sqrt(2.0/D)			# weight matrix for data flowing from input to 1st hidden layer 
b_inp_1 = np.zeros((1, h))						# bias matrix for data flowing from input to 1st hidden layer 
W_1_out = 0.01 * np.random.randn(h, K)					# weight matrix for data flowing from hidden layer to output layer 
b_1_out = np.zeros((1, K))						# bias matrix for data flowing from hidden layer to output layer  

# =======================================================================
#			initialize hyperparameters
# =======================================================================
learning_rate = 0.09			
regularization_rate = 1e-3

# =======================================================================
#			learning rate decay
# =======================================================================
decay = learning_rate/iterations
def learning_rate_annealing(learning_rate, epoch):
	return learning_rate * (1/(1 + decay * epoch))

# =======================================================================
#			gradient descent
# =======================================================================
loss = []
epochs = []
train_acc = []
valid_acc = []

for i in range(1, (iterations + 1)):

	np.random.shuffle(train_data)					# shuffle dataset
	train, valid = train_data[:50000, :], train_data[50000:, :]	# 50000 training examples, 10000 validation examples

	y_train = train[:, -1]						# select last column which is column containing labels (50000 X 1)
	X_train = np.delete(train, -1, axis = 1)			# remove last column and this becomes training input (50000 X 784)
	X_train = X_train/255.0

	y_valid = valid[:, -1]						# select last column which is column containing labels (10000 X 1)
	X_valid = np.delete(valid, -1, axis = 1)			# remove last column and this becomes training input (10000 X 784)
	X_valid = X_valid/255.0

	X_batches = np.split(X_train, 1000)				# split training set into 1000 batches with 50 records in each batch
	y_batches = np.split(y_train, 1000)				# split training set into 1000 batches with 50 records in each batch

	batch_loss = []

	# learning_rate = learning_rate_annealing(learning_rate, i)	# learning rate annealing		

	for j in range(len(X_batches)):
		X = X_batches[j]
		y = y_batches[j]
		num_records = X.shape[0]

		# variance of incoming data
 
		# ===================== forward propogation =============================
		S_1 = np.dot(X, W_inp_1) + b_inp_1		# compute (WX + b) in hidden layer
		Z_1 = np.maximum(0, S_1)			# ReLU activation function. Final output from hidden layer-1 is in Z_1
		S_2 = np.dot(Z_1, W_1_out) + b_1_out				# compute (WX + b) for outer layer
		output = np.exp(S_2)						# compute probabilities. 
		output = output/np.sum(output, axis = 1, keepdims = True) 	# Softmax function is the activation function in outer layer

		# compute total loss i.e. (total_loss = cross-entropy loss + reguralization)
		cross_entropy_loss = -np.log(output[range(num_records), y])			# taking negative logarithm
		cross_entropy_loss = np.sum(cross_entropy_loss)/num_records			# averaging over the entire sum
		reguralization = (0.5) * regularization_rate * np.sum(W_1_out * W_1_out)	# using L2 reguralization
		total_loss = cross_entropy_loss + reguralization				# total loss computation

		batch_loss.append(total_loss)
	
		# ===================== backpropogation =================================
		# back propogation follows a set of matrix multiplications. The format is simple
		# first backpropagate the changes and finally update the parameters
		# derivative/gradient of softmax function used in output layer is http://cs231n.github.io/neural-networks-case-study/#loss
		D_out = output
		D_out[range(num_records), y] = D_out[range(num_records), y] - 1		# backproping gradient into softmax function first
		D_out = D_out/num_records						# then backproping gradient into output layer 
	
		dW_1_out = np.dot(Z_1.T, D_out)						# backproping the above gradient to weights
		db_1_out = np.sum(D_out, axis = 0, keepdims = True)			# backproping the above gradient to bias
		
		dW_1_out_reg = regularization_rate * W_1_out				# gradient from L2 regularization
		
		# backpropagate the above updates to hidden layer -- 
		# derivative of ReLU function is http://kawahara.ca/what-is-the-derivative-of-relu/
		# implementing the formula dW_inp_1 = derivative(f(layer-1)) o (D_out . W_1_out.T)
		D_1 = S_1								# backproping gradient into ReLU function first
		D_1[D_1 < 0] = 0
		D_1[D_1 > 0] = 1	

		D_1 = np.multiply(D_1, np.dot(D_out, W_1_out.T))			# then backpropagating gradient into hidden layer
	
		dW_inp_1 = np.dot(X.T, D_1)						# backproping the above gradient to weights
		db_inp_1 = np.sum(D_1, axis = 0, keepdims = True)			# backproping the above gradient to bias

		dW_inp_1_reg = regularization_rate * W_inp_1				# gradient from L2 regularization

		# update W_1_out as W_1_out = W_1_out - learning_rate * [gradient of weights + gradient of reguralization]
		# update W_in_1 as W_in_1 = W_in_1 - learning_rate * [gradient of weights + gradient of reguralization] 
		# update b_1_out and b_2_out as well
		
		W_1_out = W_1_out - learning_rate * (dW_1_out + dW_1_out_reg)
		W_inp_1 = W_inp_1 - learning_rate * (dW_inp_1 + dW_inp_1_reg)

		b_1_out = b_1_out - learning_rate * db_1_out
		b_inp_1 = b_inp_1 - learning_rate * db_inp_1


	epochs.append(i+1)
	loss.append(np.mean(batch_loss))
	
	# =======================================================================
	#			training set accuracy
	# =======================================================================
	S_1 = np.dot(X_train, W_inp_1) + b_inp_1		# compute (WX + b) on hidden layer
	Z_1 = np.maximum(0, S_1)			# ReLU activation function. Final output from hidden layer-1 is stored in Z_1
	S_2 = np.dot(Z_1, W_1_out) + b_1_out		# compute (WX + b) for outer layer
	prediction = np.argmax(S_2, axis = 1)		# check along every row which class label has max. score
	mean = np.mean(prediction == y_train)
	print "iteration: ", i, " training accuracy: ", mean
	train_acc.append(mean)

	# =======================================================================
	#			Validation set accuracy
	# =======================================================================
	S_1 = np.dot(X_valid, W_inp_1) + b_inp_1	# compute (WX + b) on hidden layer
	Z_1 = np.maximum(0, S_1)			# ReLU activation function. Final output from hidden layer-1 is stored in Z_1
	S_2 = np.dot(Z_1, W_1_out) + b_1_out		# compute (WX + b) for outer layer
	prediction = np.argmax(S_2, axis = 1)		# check along every row which class label has max. score
	mean = np.mean(prediction == y_valid)
	print "iteration: ", i, " validation accuracy: ", mean
	valid_acc.append(mean)

print time.clock() - start_time, "seconds"

# plotting loss curve
fig1 = plt.figure()
plt.plot(epochs, loss)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show(fig1)

# plotting training and validation accuracies
fig2 = plt.figure()
plt.plot(epochs, train_acc, 'r')
plt.plot(epochs, valid_acc, 'g')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.show(fig2)

# =======================================================================
#				test dataset
# =======================================================================
test_images = idx2numpy.convert_from_file('/home/karthikeya/Desktop/nn/t10k-images.idx3-ubyte')
test_labels = idx2numpy.convert_from_file('/home/karthikeya/Desktop/nn/t10k-labels.idx1-ubyte')

# converting test dataset from (10000 X 28 X 28) to (10000 X 784)
test = []
for i in range(len(test_images)):
	test.append(test_images[i].flatten())

test = np.asarray(test)						# test dataset of dimensiosn (10000 X 784)
test = test/255.0	

# compute layer wise outputs
S_1 = np.dot(test, W_inp_1) + b_inp_1				# compute (WX + b) on hidden layer
Z_1 = np.maximum(0, S_1)					# ReLU activation function. Output from hidden layer-1 is stored in Z_1
S_2 = np.dot(Z_1, W_1_out) + b_1_out				# compute (WX + b) for outer layer
prediction = np.argmax(S_2, axis = 1)				# check along every row which class label has max. score
mean = np.mean(prediction == test_labels)
print "test accuracy: ", mean



