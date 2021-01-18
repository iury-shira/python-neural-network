import numpy as np

X = [
	[1.0, 2.0, 3.0, 2.5],
	[2.0, 5.0, -1.0, 2.0],
	[-1.5, 2.7, 3.3, -0.8]
]


class LayerDense:
	def __init__(self, n_inputs, n_neurons):
		self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
		self.biases = np.zeros([1, n_neurons])
		self.output = []

	def forward(self, inputs):
		self.output = np.dot(inputs, self.weights) + self.biases


class ActivationReLu:
	def __init__(self):
		self.outputs = np.ndarray

	def forward(self, inputs):
		self.outputs = np.maximum(0, inputs)


layer1 = LayerDense(4, 5)
layer2 = LayerDense(5, 2)

layer1.forward(X)
layer2.forward(layer1.output)
print(layer2.output)

print('\n\n')

activation = ActivationReLu()
activation.forward(layer2.output)
print(activation.outputs)

'''
The numpy.random.randn() function creates an array of specified shape and fills it with random values 
as per standard normal distribution. The random values are floats sampled from a univariate “normal” (Gaussian) 
distribution of mean 0 and variance 1.


'''