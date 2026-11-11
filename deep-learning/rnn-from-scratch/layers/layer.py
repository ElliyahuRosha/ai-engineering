class Layer:
	def __init__(self):
		self.input = None
		self.output = None

	def forward(self, input):
		# will return output
		pass

	def backward(self, dE_dy, learning_rate):
		# will update weights and return dE_dx
		pass
