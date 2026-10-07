import numpy as np
from .layer import Layer

class Activation(Layer):
    def __init__(self, func, func_deriv):
        self.func = func
        self.func_deriv = func_deriv

    def forward(self, input):
        self.input = input
        self.output = self.func(self.input)                         # f(X)
        return self.output

    def backward(self, dE_dy, learning_rate):
        dE_dx = np.multiply(dE_dy, self.func_deriv(self.input))     # E' ⊙ f'(X)
        return dE_dx
