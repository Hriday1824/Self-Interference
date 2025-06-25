# -*- coding: utf-8 -*-
# """
# Created on Wed Jun 29 12:42:24 2022

# @author: admin
# Modified for Spline Adaptive Filter approach
# """

import scipy.io
from scipy.signal import savgol_filter
import numpy as np
import fullduplexmvc as fd
import matplotlib.pyplot as plt
import os

# Disable GPU use if needed
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Define system parameters
params = {
    'samplingFreqMHz': 20,      # Sampling frequency
    'hSILen': 13,              # Self-interference channel length
    'pamaxordercanc': 7,       # Maximum PA non-linearity order
    'trainingRatio': 0.9,      # Ratio of total samples to use for training
    'dataOffset': 14,          # Data offset for transmitter-receiver misalignment
    'splineControlPoints': 300, # Number of control points for spline interpolation
    'splineLearningRate': 0.001,# Learning rate for spline adaptation
    'filterLearningRate': 0.001,# Learning rate for linear filter adaptation
}

##### Load and prepare data #####
x, y, noise, measuredNoisePower = fd.loadData('sample.mat', params)

##### Spline Adaptive Filter Implementation #####
class SplineAdaptiveFilter:
    def __init__(self, filter_length, num_control_points, spline_lr, filter_lr):
        self.filter_length = filter_length
        self.num_control_points = num_control_points
        self.spline_lr = spline_lr
        self.filter_lr = filter_lr

        # Initialize linear filter coefficients
        self.w = np.zeros(filter_length, dtype=np.complex128)

        # Initialize spline control points
        self.control_points_x = np.linspace(-1, 1, num_control_points)
        self.control_points_y = np.linspace(-1, 1, num_control_points)+1j * np.linspace(-3, 3, num_control_points)

    def spline_interpolation(self, x):
        # Restrict interpolation input range
        x_real = np.clip(x.real, self.control_points_x[0], self.control_points_x[-1])
        idx = np.searchsorted(self.control_points_x, x_real) - 1
        # idx = np.clip(idx, 0, self.num_control_points - 2)

        denom = self.control_points_x[idx + 1] - self.control_points_x[idx]
        t = (x_real - self.control_points_x[idx]) / denom if denom != 0 else 0
        # t = np.clip(t, 0, 1)

        p0 = self.control_points_y[idx]
        p1 = self.control_points_y[idx + 1]

        return p0 + t * (p1 - p0)

    def spline_derivative(self, x):
        x_real = np.clip(x.real, self.control_points_x[0], self.control_points_x[-1])
        idx = np.searchsorted(self.control_points_x, x_real) - 1
        # idx = np.clip(idx, 0, self.num_control_points - 2)

        denom = self.control_points_x[idx + 1] - self.control_points_x[idx]
        return (self.control_points_y[idx + 1] - self.control_points_y[idx]) / denom if denom != 0 else 0

    def update_spline_control_points(self, x, error):
        x_real = np.clip(x.real, self.control_points_x[0], self.control_points_x[-1])
        idx = np.searchsorted(self.control_points_x, x_real) - 1
        # idx = np.clip(idx, 0, self.num_control_points - 2)

        denom = self.control_points_x[idx + 1] - self.control_points_x[idx]
        t = (x_real - self.control_points_x[idx]) / denom if denom != 0 else 0
        # t = np.clip(t, 0, 1)

        update = self.spline_lr * error
        # max_update = 0.05
        # if np.abs(update) > max_update:
        #     update = update / np.abs(update) * max_update

        self.control_points_y[idx] += (1 - t) * update
        self.control_points_y[idx + 1] += t * update

    def predict(self, x_window):
        linear_output = np.dot(self.w.conj(), x_window)
        return self.spline_interpolation(linear_output)

    def update(self, x_window, error):
        norm = np.linalg.norm(x_window)
        # if norm > 1e-6:
        #     x_window = x_window / norm

        linear_output = np.dot(self.w.conj(), x_window)
        # linear_output_real = np.clip(linear_output.real, self.control_points_x[0], self.control_points_x[-1])
        # linear_output_imag = np.clip(linear_output.imag, self.control_points_x[0], self.control_points_x[-1])
        linear_output = complex(linear_output.real, linear_output.imag)

        spline_derivative = self.spline_derivative(linear_output)
        grad = self.filter_lr * np.conj(error * spline_derivative) * x_window
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1.0:
            grad = grad / grad_norm

        self.w += grad
        self.update_spline_control_points(linear_output, error)


##### Training #####
# Step 1: Estimate linear cancellation parameters and perform linear cancellation
trainingSamples = int(np.floor(x.size*params['trainingRatio']))
x_train = x[0:trainingSamples]
y_train = y[0:trainingSamples]
x_test = x[trainingSamples:]
y_test = y[trainingSamples:]

hLin = fd.SIestimationLinear(x_train, y_train, params)
yCanc = fd.SIcancellationLinear(x_train, hLin, params)

# Normalize data for SAF
y_train = y_train - yCanc
yVar = np.var(y_train)
y_train = y_train/np.sqrt(yVar)

# Initialize SAF
saf = SplineAdaptiveFilter(
    filter_length=params['hSILen'],
    num_control_points=params['splineControlPoints'],
    spline_lr=params['splineLearningRate'],
    filter_lr=params['filterLearningRate']
)

# Prepare training data
chanLen = params['hSILen']
x_train_windows = np.array([x_train[i:i+chanLen] for i in range(x_train.size-chanLen)])
y_train_target = y_train[chanLen:]

# Training loop
for epoch in range(30):  # Number of training passes
    total_error = 0
    for i in range(len(x_train_windows)):
        # Predict output
        y_pred = saf.predict(x_train_windows[i])

        # Compute instantaneous error
        error = y_train_target[i] - y_pred

        # Accumulate squared error for monitoring
        total_error += (error)**2

        # Update SAF with this sample
        saf.update(x_train_windows[i], error)
    
    # Optional: Print average MSE for this epoch
    mse = total_error / len(x_train_windows)
    print(f"Epoch {epoch+1}, MSE: {mse:.6f}")

##### Test #####
# Prepare test data
yCanc = fd.SIcancellationLinear(x_test, hLin, params)
yOrig = y_test
y_test = y_test - yCanc
y_test = y_test/np.sqrt(yVar)

x_test_windows = np.array([x_test[i:i+chanLen] for i in range(x_test.size-chanLen)])
y_test_target = y_test[chanLen:]

# Perform non-linear cancellation with SAF
yCancNonLin = np.zeros(len(x_test_windows), dtype=np.complex128)
for i in range(len(x_test_windows)):
    yCancNonLin[i] = saf.predict(x_test_windows[i])

# Scale back to original power
# yCancNonLin = yCancNonLin * np.sqrt(yVar)

##### Evaluation #####
y_test = yOrig[chanLen:]
yCanc = yCanc[chanLen:]

# Calculate various signal powers
noisePower = 10*np.log10(np.mean(np.abs(noise)**2))
scalingConst = np.power(10,-(measuredNoisePower-noisePower)/10)
noise /= np.sqrt(scalingConst)
y_test /= np.sqrt(scalingConst)
yCanc /= np.sqrt(scalingConst)
yCancNonLin /= np.sqrt(scalingConst)

# Plot PSD and get signal powers
noisePower, yTestPower, yTestLinCancPower, yTestNonLinCancPower = fd.plotPSD(
    y_test, yCanc, yCancNonLin, noise, params, 'SAF', yVar)

# Print cancellation performance
print('')
print('The linear SI cancellation is: {:.6f} dB'.format(yTestPower-yTestLinCancPower))
print('The non-linear SI cancellation is: {:.6f} dB'.format(yTestLinCancPower-yTestNonLinCancPower))
print('The noise floor is: {:.6f} dBm'.format(noisePower))
print('The distance from noise floor is: {:.6} dB'.format(yTestNonLinCancPower-noisePower))

# Plot spline characteristic
plt.figure()
plt.plot(saf.control_points_x, saf.control_points_y.real, 'bo-')
plt.plot(saf.control_points_x, saf.control_points_y.imag, 'ro-')
plt.title('Learned Spline Characteristic')
plt.xlabel('Input')
plt.ylabel('Output')
plt.legend(['Real part', 'Imaginary part'])
plt.grid(True)
plt.show()