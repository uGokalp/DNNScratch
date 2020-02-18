import numpy as np
from layers import HiddenLayer
from activations import sigmoid, relu, sigmoid_derivative


class NN:

    def __init__(self, X, y):
        self.hidden = []
        self.X = X
        self.y = y
        self.n = len(y)

    def create_layers(self, num_hidden, in_dim, out_dim, n_class):
        for n in range(num_hidden - 1):
            _ = HiddenLayer(in_dim, out_dim)
            self.hidden.append(_)
        self.hidden.append(HiddenLayer(out_dim, n_class))

    def forward(self, x):
        for layer in self.hidden:
            x = layer.forward(x)
        return x

    def backward(self):
        self.hidden_deltas = []
        expected = self.y

        # calc delta for output layer
        self.output_delta = (self.hidden[-1].f[
                                 'W'] - expected) * sigmoid_derivative(
            self.hidden[-1].f['W'])
        hidden_delta = self.output_delta

        # Calculate delta for others
        self.hidden_delta = np.dot(self.output_delta,
                                   self.hidden[
                                       1].weight.T) * sigmoid_derivative(
            self.hidden[0].f["W"])

    def update(self, lr):
        """
        Updates using Mean Squared Loss Error
        :return:
        """
        self.hidden[1].weight -= lr * np.dot(self.hidden[1].f["W"].T,
                                             self.output_delta) / self.n
        self.hidden[0].weight -= lr * np.dot(self.X.T,
                                             self.hidden_delta) / self.n

    def train(self, lr, epochs):
        for epoch in range(epochs):
            self.forward(self.X)
            self.backward()
            self.update(lr)
