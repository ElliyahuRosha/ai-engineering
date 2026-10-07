import numpy as np

from layers.activation import Activation
from layers.layer import Layer

class Tanh(Activation):
    def __init__(self):
        def tanh(x):
            return np.tanh(x)

        def tanh_deriv(x):
            return 1 - np.tanh(x) ** 2

        super().__init__(tanh, tanh_deriv)

class Sigmoid(Activation):
    def __init__(self):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def sigmoid_deriv(x):
            s = sigmoid(x)
            return s * (1 - s)

        super().__init__(sigmoid, sigmoid_deriv)
