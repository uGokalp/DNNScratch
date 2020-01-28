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


def relu(x):
    return np.maximum(0, x)
