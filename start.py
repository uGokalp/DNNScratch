import numpy as np


class Model:
    """
    What it should do:

    """

    def __init__(self, n_layers, input_size, output_size):
        self.n_layers = n_layers
        self.layers = []
        self.input_size = input_size
        self.output_size = output_size

    def push_layer(self, new_layer):
        self.layers.append(new_layer)

    def train(self, X_train, y_train):
        if not self.layers:
            pass
        else:
            print("Need layers fam")

    def test(self, X_test, y_test):
        pass


class Dataset:
    """
    Should load data in batches
    :return:
    """
