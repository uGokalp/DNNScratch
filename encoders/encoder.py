import numpy as np


class Encoder:
    """
    Base class for encoders.
    """

    def __init__(self):
        pass

    def fit(self, X):
        raise NotImplemented

    def transform(self, X):
        raise NotImplemented

    def fit_transform(self, X: np.ndarray):
        self.fit(X)
        return self.transform(X)
