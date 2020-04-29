from encoder import Encoder
import numpy as np
from label_encoder import LabelEncoder


class MeanEncoder(Encoder):
    """
    Only support for binary.
    ToDo: Support for multi-class.
    """
    def __init__(self):
        self.mapper = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        unique = np.unique(X)
        mapper = dict().fromkeys(unique, 0)
        for k in unique:
            mapper[k] = y[X == k].mean()
        self.mapper = mapper

    def transform(self, X):
        placeholder = X.copy()
        return np.array(map(lambda x: self.mapper[x], placeholder))
