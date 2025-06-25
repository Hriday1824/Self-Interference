import scipy.io
from scipy.signal import savgol_filter
import numpy as np
import fullduplexlms as fd
from keras.models import Model, Sequential 
from keras.layers import Dense, Input, SimpleRNN, Dropout
import tensorflow.keras
#from keras.optimizers import Adam
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import os 



# This line disables the use of the GPU for training. The dataset is not large enough to get
# significant gains from GPU training and, in fact, sometimes training can even be slower on
# the GPU than on the CPU. Comment out to enable GPU use.
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Define system parameters
params = {
		'samplingFreqMHz': 20,	# Sampling frequency, required for correct scaling of PSD
		'hSILen': 13,			# Self-interference channel length
		'pamaxordercanc': 7,	# Maximum PA non-linearity order
		'trainingRatio': 0.9,	# Ratio of total samples to use for training
		'dataOffset': 14,		# Data offset to take transmitter-receiver misalignment into account
		'nHidden': 17,			# Number of hidden layers in NN
		'nEpochs': 20,			# Number of training epochs for NN training
		'learningRate': 0.004,	# Learning rate for NN training
		'batchSize': 32,		# Batch size for NN training
		}

##### Load and prepare data #####

x, y, noise, measuredNoisePower = fd.loadData('sample.mat', params)
h=np.array([-0.00040268-0.00018969j,  0.00180184+0.00072207j,
       -0.0053656 -0.00196921j,  0.01588391+0.00553579j,
       -0.14671433-0.05495991j, -0.03969207-0.00427781j,
        0.01919757+0.00672024j, -0.01149634-0.00510535j,
        0.00894478+0.00379464j, -0.00656337-0.00267423j,
        0.00398431+0.00160615j, -0.00189671-0.00074281j,
        0.00051147+0.00020373j])
#h=1/h;
trainingSamples = int(np.floor(x.size*params['trainingRatio'])); 
# Get self-interference channel length
chanLen = params['hSILen']; 
x_train = x[0:trainingSamples]
#x1=np.convolve(x+r, h, mode='same')
# Create feedforward NN using Keras
nHidden = params['nHidden']
nEpochs = params['nEpochs']
input = Input(shape=(2*chanLen,))
hidden1 = Dense(nHidden, activation='relu')(input)
output1 = Dense(1, activation='linear')(hidden1)
output2 = Dense(1, activation='linear')(hidden1)
model = Model(inputs=input, outputs=[output1, output2])
adam = Adam(learning_rate=params['learningRate'])
model.compile(loss = "mse", optimizer = adam)
print("Total number of real parameters to estimate for neural network based canceller: {:d}".format((2*chanLen+1)*nHidden + 2*(nHidden+1)+2*chanLen)); x_train[5000]=10
outlier=(20+10*1j)*np.ones([1,18412])
# Split into training and test sets
trainingSamples = int(np.floor(x.size*params['trainingRatio'])); 
x_train = x[0:trainingSamples]
#x_train[22]=10+x_train[22]; x_train[1000]=1+x_train[1000]; x_train[2200]=1+x_train[2200]; x_train[8000]=1+x_train[8000]; 
#x_train = x_train+
y_train = y[0:trainingSamples]
#y_train=y_train-x[0:trainingSamples]
#y_train = y[0:trainingSamples]
x_test = x[trainingSamples:]
#x_test[trainingSamples-19000]=1+x_test[trainingSamples-19000]; x_test[trainingSamples-19500]=10+x_test[trainingSamples-19500];
y_test = y[trainingSamples:]#+outlier[0,1000]
#y_test=y_test-x[trainingSamples:]
#y_test = y[trainingSamples:]

##### Training #####
# Step 1: Estimate linear cancellation arameters and perform linear cancellation
#x_train[5000]=1000
#x_train[10000]=0
hLin = fd.SIestimationLinear(x_train, y_train, params)# fd.SIestimationLinear(x_train, y_train, params)
#hLin=h
x_train[5000]=1000
x_train[10000]=400
yCanc = fd.SIcancellationLinear(x_train, hLin, params)# fd.SIcancellationLinear(x_train, hLin, params)

# Normalize data for NN
yOrig = y_train
y_train = y_train - yCanc
yVar = np.var(y_train)
y_train = y_train/np.sqrt(yVar)

# Prepare training data for NN
x_train_real = np.reshape(np.array([x_train[i:i+chanLen].real for i in range(x_train.size-chanLen)]), (x_train.size-chanLen, chanLen))
x_train_imag = np.reshape(np.array([x_train[i:i+chanLen].imag for i in range(x_train.size-chanLen)]), (x_train.size-chanLen, chanLen))
x_train = np.zeros((x_train.size-chanLen, 2*chanLen))
x_train[:,0:chanLen] = x_train_real
x_train[:,chanLen:2*chanLen] = x_train_imag
y_train = np.reshape(y_train[chanLen:], (y_train.size-chanLen, 1))

# Prepare test data for NN
yCanc = fd.SIcancellationLinear(x_test, hLin, params)
yOrig = y_test
y_test = y_test - yCanc
y_test = y_test/np.sqrt(yVar)

x_test_real = np.reshape(np.array([x_test[i:i+chanLen].real for i in range(x_test.size-chanLen)]), (x_test.size-chanLen, chanLen))
x_test_imag = np.reshape(np.array([x_test[i:i+chanLen].imag for i in range(x_test.size-chanLen)]), (x_test.size-chanLen, chanLen))
x_test = np.zeros((x_test.size-chanLen, 2*chanLen))
x_test[:,0:chanLen] = x_test_real
x_test[:,chanLen:2*chanLen] = x_test_imag
y_test = np.reshape(y_test[chanLen:], (y_test.size-chanLen, 1))

##### Training #####
# Step 2: train NN to do non-linear cancellation
history = model.fit(x_train, [y_train.real, y_train.imag], epochs = nEpochs, batch_size = params['batchSize'], verbose=2, validation_data= (x_test, [y_test.real, y_test.imag]))

##### Test #####
# Do inference step
pred = model.predict(x_test)
yCancNonLin = np.squeeze(pred[0] + 1j*pred[1], axis=1)

##### Evaluation #####
# Get correctly shaped test and cancellation data
y_test = yOrig[chanLen:]
yCanc = yCanc[chanLen:]

# Calculate various signal powers
noisePower = 10*np.log10(np.mean(np.abs(noise)**2))
scalingConst = np.power(10,-(measuredNoisePower-noisePower)/10)  #measuredNoisePower
noise /= np.sqrt(scalingConst)
y_test /= np.sqrt(scalingConst)
yCanc /= np.sqrt(scalingConst)
yCancNonLin /= np.sqrt(scalingConst)

# Plot PSD and get signal powers
noisePower, yTestPower, yTestLinCancPower, yTestNonLinCancPower = fd.plotPSD(y_test, yCanc, yCancNonLin, noise, params, 'NN', yVar)

# Print cancellation performance
print('')
print('The linear SI cancellation is: {:.2f} dB'.format(yTestPower-yTestLinCancPower))
print('The non-linear SI cancellation is: {:.2f} dB'.format(yTestLinCancPower-yTestNonLinCancPower))
print('The noise floor is: {:.2f} dBm'.format(noisePower))
print('The distance from noise floor is: {:.2f} dB'.format(yTestNonLinCancPower-noisePower))

# Plot learning curve
plt.plot(np.arange(1,len(history.history['loss'])+1), -10*np.log10(history.history['loss']), 'bo-')
plt.plot(np.arange(1,len(history.history['loss'])+1), -10*np.log10(history.history['val_loss']), 'ro-')
plt.ylabel('Self-Interference Cancellation (dB)')
plt.xlabel('Training Epoch')
plt.legend(['Training Frame', 'Test Frame'], loc='lower right')
plt.grid(which='major', alpha=0.25)
plt.xlim([ 0, nEpochs+1 ])
plt.xticks(range(1,nEpochs,2))
#plt.savefig('figures/NNconv.pdf', bbox_inches='tight')
plt.show()

# class SplineAdaptiveFilter:
#     def __init__(self, n_knots, learning_rate, input_dim):
#         """
#         Initialize the Spline Adaptive Filter.

#         Args:
#             n_knots (int): Number of knots for the spline.
#             learning_rate (float): Learning rate for gradient descent.
#             input_dim (int): Dimensionality of the input (2*chanLen in your case).
#         """
#         self.n_knots = n_knots
#         self.learning_rate = learning_rate
#         self.input_dim = input_dim

#         # Initialize spline control points (knots)
#         self.control_points1 = np.random.randn(n_knots)  # For output 1
#         self.control_points2 = np.random.randn(n_knots)  # For output 2

#         # Initialize linear weights
#         self.weights1 = np.random.randn(input_dim)  # For output 1
#         self.weights2 = np.random.randn(input_dim)  # For output 2

#         # Spline grid (uniformly spaced)
#         self.spline_grid = np.linspace(-1, 1, n_knots)

#     def spline_interpolation(self, x, control_points):
#         """
#         Perform spline interpolation.

#         Args:
#             x (float): Input value.
#             control_points (np.array): Control points (knots) for the spline.

#         Returns:
#             float: Interpolated value.
#         """
#         # Clip x to the spline grid range to avoid extrapolation
#         x = np.clip(x, self.spline_grid[0], self.spline_grid[-1])
        
#         # Find the nearest knots
#         idx = np.searchsorted(self.spline_grid, x) - 1
#         idx = max(0, min(idx, self.n_knots - 2))  # Ensure within bounds

#         # Linear interpolation between knots
#         x0, x1 = self.spline_grid[idx], self.spline_grid[idx + 1]
#         y0, y1 = control_points[idx], control_points[idx + 1]
        
#         if np.isclose(x1, x0):
#             return y0
#         return y0 + (y1 - y0) * (x - x0) / (x1 - x0)

#     def forward(self, x):
#         """
#         Forward pass of the SAF.

#         Args:
#             x (np.array): Input vector of shape (input_dim,).

#         Returns:
#             tuple: Outputs of the SAF (output1, output2).
#         """
#         # Linear transformation
#         linear_output1 = np.dot(x, self.weights1)
#         linear_output2 = np.dot(x, self.weights2)

#         # Spline transformation
#         spline_output1 = self.spline_interpolation(linear_output1, self.control_points1)
#         spline_output2 = self.spline_interpolation(linear_output2, self.control_points2)

#         return spline_output1, spline_output2

#     def train(self, X, y1, y2, epochs):
#         """
#         Train the SAF using gradient descent.

#         Args:
#             X (np.array): Input data of shape (n_samples, input_dim).
#             y1 (np.array): Target output 1 of shape (n_samples,).
#             y2 (np.array): Target output 2 of shape (n_samples,).
#             epochs (int): Number of training epochs.
#         """
#         for epoch in range(epochs):
#             for i in range(X.shape[0]):
#                 x = X[i]
#                 target1, target2 = y1[i], y2[i]

#                 # Forward pass
#                 output1, output2 = self.forward(x)

#                 # Compute errors
#                 error1 = output1 - target1
#                 error2 = output2 - target2
                
#                 # Clip errors to avoid overflow
#                 error1 = np.clip(error1, -1e6, 1e6)
#                 error2 = np.clip(error2, -1e6, 1e6)

#                 # Backpropagation (gradient computation)
#                 # Gradient for linear weights
#                 grad_weights1 = error1 * x
#                 grad_weights2 = error2 * x

#                 # Gradient for spline control points
#                 grad_control_points1 = error1 * self.spline_gradient(output1, self.control_points1)
#                 grad_control_points2 = error2 * self.spline_gradient(output2, self.control_points2)

#                 # Update parameters
#                 self.weights1 -= self.learning_rate * grad_weights1
#                 self.weights2 -= self.learning_rate * grad_weights2
#                 self.control_points1 -= self.learning_rate * grad_control_points1
#                 self.control_points2 -= self.learning_rate * grad_control_points2

#             if epoch % 100 == 0 and i == 0:
#                 print(f"Epoch {epoch}, Sample {i}:")
#                 print(f"  x: {x}")
#                 print(f"  output1: {output1}, target1: {target1}, error1: {error1}")
#                 print(f"  output2: {output2}, target2: {target2}, error2: {error2}")

#     def spline_gradient(self, x, control_points):
#         """
#         Compute the gradient of the spline interpolation with respect to control points.

#         Args:
#             x (float): Input value.
#             control_points (np.array): Control points (knots) for the spline.

#         Returns:
#             np.array: Gradient of the spline interpolation.
#         """
#         # Clip x to the spline grid range to avoid extrapolation
#         x = np.clip(x, self.spline_grid[0], self.spline_grid[-1])
        
#         idx = np.searchsorted(self.spline_grid, x) - 1
#         idx = max(0, min(idx, self.n_knots - 2))  # Ensure within bounds

#         grad = np.zeros_like(control_points)
#         x0, x1 = self.spline_grid[idx], self.spline_grid[idx + 1]
#         if np.isclose(x0, x1):
#             grad[idx] = 0.5
#             grad[idx + 1] = 0.5
#         else:
#             grad[idx] = (x1 - x) / (x1 - x0)
#             grad[idx + 1] = (x - x0) / (x1 - x0)
#         return grad
