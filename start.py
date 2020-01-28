import numpy as np


def sigmoid(x):
    """
    Sigmoid(x) = 1 / (1 + e^-x)
    :param x:
    :return:
    """
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


class Dense:
    def __init__(self, inputs, outputs):
        self.input = inputs
        self.output = outputs
        self.bias = self._get_bias()

    def _get_bias(self):
        in_dim = np.ndim(self.input)
        out_dim = np.ndim(self.output)
        return np.random.random((in_dim, out_dim))


class Model:
    def __init__(self, n_layers, input_size, output_size):
        self.n_layers = n_layers
        self.layers = []
        self.input_size = input_size

    def agg_layers(self):
        if len(self.layers) < 1:
            for layer in range(self.n_layers):
                self.layers.append(Dense)

    def solve_shapes(self):
        pass

    def Dataset(self):
        """
        Should load data in batches
        :return:
        """
