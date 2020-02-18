import numpy as np


def sigmoid(x):
    """
    Sigmoid(x) = 1 / (1 + e^-x)
    :param x:
    :return:
    """
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    """
    Expecting x to already have sigmoid applied
    :param x:
    :return:
    """
    return x * (1 - x)


def relu(x):
    return np.maximum(0, x)


##################
#### METRICS #####
##################


def mse(y_true, pred):
    error = np.square((y_true - pred)).sum() / 2 * len(y_true)
    return error


def accuracy(y_true, pred):
    corrects = pred.round() == y_true
    return corrects.mean()
