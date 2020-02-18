import numpy as np


def xaview_init(size: tuple, input, output):
    """
    Xaview weight initialization
    :param size:
    :return:
    """
    return np.random.randint((size)) * np.sqrt(1 / (input + output))


def he_init(size: tuple, input, output):
    return 2 * xaview_init(size, input, output)
