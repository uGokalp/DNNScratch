import numpy as np


class Dense:
    def __init__(self, inputs, outputs):
        self.input = inputs
        self.output = outputs
        self.bias = self._get_bias()

    def _get_bias(self):
        in_dim = np.ndim(self.input)
        out_dim = np.ndim(self.output)
        return np.random.random((in_dim, out_dim))
