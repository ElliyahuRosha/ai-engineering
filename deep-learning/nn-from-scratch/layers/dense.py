import numpy as np
from .layer import Layer

class Dense(Layer):
	def __init__(self, input_size, output_size):
		self.weights = np.random.randn(output_size, input_size)		# W matrix
		self.bias = np.random.randn(output_size, 1)					# B vector

	def forward(self, input):
		self.input = input
		self.output = np.dot(self.weights, self.input) + self.bias  	# W @ X + B
		return self.output

	def backward(self, dE_dy, learning_rate):
		dE_dw = np.dot(dE_dy, self.input.T)							# W_grad = E' @ X^T
		dE_dx = np.dot(self.weights.T, dE_dy)						# X_grad = W^T @ E'

		# Update params
		self.weights	-= learning_rate * dE_dw
		self.bias		-= learning_rate * dE_dy					# B_grad = E'

		# return X_grad
		return dE_dx
