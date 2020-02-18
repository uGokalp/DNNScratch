import numpy as np
from activations import sigmoid
from random import random


class HiddenLayer:

    def __init__(self, in_dim, out_dim):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.init_layer()
        self.f = {}

    def init_layer(self):
        self.weight = np.random.rand(self.in_dim, self.out_dim)
        self.bias = np.random.rand(1, self.out_dim)

    def forward(self, x):
        if self.weight.shape[0] == x.shape[1]:
            x = np.dot(x, self.weight) + self.bias
            x = sigmoid(x)
            self.f['W'] = x
            return x
        else:
            x = sigmoid(np.matmul(x, self.weight) + self.bias)
            self.f['W'] = x
            return x
