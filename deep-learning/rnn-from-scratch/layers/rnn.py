import numpy as np
from .layer import Layer

class Rnn(Layer):
	# def __init__(self, input_size, hidden_size):
	# 	self.weights_x = np.random.randn(hidden_size, input_size)		# Wx matrix
	# 	self.weights_h = np.random.randn(hidden_size, hidden_size)		# Wh matrix
	# 	self.bias = np.random.randn(hidden_size, 1)						# B vector
	# 	self.hidden_size = hidden_size

	def __init__(self, input_size, hidden_size):
		self.weights_x = np.random.randn(hidden_size, input_size) / np.sqrt(input_size)	# Wx matrix
		self.weights_h = np.random.randn(hidden_size, hidden_size) / np.sqrt(hidden_size)	# Wh matrix
		self.bias = np.zeros((hidden_size, 1))												# B vector
		self.hidden_size = hidden_size

	def forward(self, input):
		self.input = input
		self.hidden_states = []							# [h0, h1, h2, ... , ht]	(t = len(input))
		self.h = np.zeros((self.hidden_size, 1))		# h0 = (0,0,...,0)
		self.hidden_states.append(self.h)

		for x in self.input:
			self.h = np.tanh(np.dot(self.weights_x, x) + np.dot(self.weights_h, self.h) + self.bias)  	# h[t] = tanh( Wx @ x[t] + Wh @ h[t-1] + B )
			self.hidden_states.append(self.h)															#			   ---------------------------
																										#				            z[t]
		self.output = self.hidden_states[-1]
		return self.output

	def backward(self, dE_dh, learning_rate):
		t = len(self.input)
		dE_dw_x = np.zeros_like(self.weights_x)		# Wx_grad accumulator
		dE_dw_h = np.zeros_like(self.weights_h)		# Wh_grad accumulator
		dE_db = np.zeros_like(self.bias)			# B_grad accumulator
		dE_dx = np.zeros_like(self.input)

		for k in range(t, 0, -1):
			dE_dz = np.multiply(dE_dh, 1 - np.multiply(self.hidden_states[k], self.hidden_states[k]))	# z_grad = h_grad ⊙ (1 - h[k]^2)
			dE_dh = np.dot(self.weights_h.T, dE_dz)														# h_prev_grad = Wh^T @ z_grad
			dE_dx[k - 1] = np.dot(self.weights_x.T, dE_dz)  											# x_grad[k] = Wx^T @ z_grad[k]

			dE_dw_x += np.dot(dE_dz, self.input[k-1].T)													# Wx_grad += z_grad @ x[k]^T
			dE_dw_h += np.dot(dE_dz, self.hidden_states[k-1].T)											# Wh_grad += z_grad @ h[k-1]^T
			dE_db   += dE_dz																			# B_grad  += z_grad

		# Store gradients for numerical verification
		self.dE_dw_x = dE_dw_x.copy()
		self.dE_dw_h = dE_dw_h.copy()
		self.dE_db = dE_db.copy()

		# Update params
		self.weights_x	-= learning_rate * dE_dw_x
		self.weights_h	-= learning_rate * dE_dw_h
		self.bias		-= learning_rate * dE_db

		# return X_grad
		return dE_dx
