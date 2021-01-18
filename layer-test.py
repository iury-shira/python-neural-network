import numpy as np

X = [
	[1.0, 2.0, 3.0, 2.5],
	[2.0, 5.0, -1.0, 2.0],
	[-1.5, 2.7, 3.3, -0.8]
]


class LayerDense:
	def __init__(self, n_inputs, n_neurons):
		self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
		self.biases = np.zeros((1, n_neurons))
		self.output = np.ndarray

	def forward(self, inputs):
		self.output = np.dot(inputs, self.weights) + self.biases


class ActivationReLu:
	def __init__(self):
		self.outputs = np.ndarray

	def forward(self, inputs):
		self.outputs = np.maximum(0, inputs)


class ActivationSoftMax:
	def __init__(self):
		self.outputs = np.ndarray

	def forward(self, inputs):
		exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
		probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
		self.outputs = probabilities


dense1 = LayerDense(4, 5)
activation1 = ActivationReLu()

dense2 = LayerDense(5, 2)
activation2 = ActivationSoftMax()

dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.outputs)
activation2.forward(dense2.output)

print(activation2.outputs)

'''
The numpy.random.randn() function creates an array of specified shape and fills it with random values 
as per standard normal distribution. The random values are floats sampled from a univariate (variance 1) 
“normal” (Gaussian) distribution of mean 0.

The numpy.zeros(shape, dtype=float, order='C), where shape can be an int or a tuple of ints, return a new
array with the given shape filled with zeros.

The numpy.dot(a, b) returns:
--> If a and b are 1-D arrays, returns its inner product
--> If a and b are 2-D arrays, returns its matrix multiplication
--> If a is a N-D array and b is a 1-D array, returns the sum product over the last axis of a and b

The numpy.maximum() compare two arrays and returns a new array containing the element-wise maxima.

The numpy.max() returns the maximum of an array or maximum along a specified axis.

'''
