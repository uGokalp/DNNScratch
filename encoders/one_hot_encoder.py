from encoder import Encoder
from label_encoder import LabelEncoder
import numpy as np


class OneHotEncoder(Encoder):

    def __init__(self):
        self.shape = None

    def fit(self, X: np.ndarray):
        unique = np.unique(X)
        n_unique = len(unique)
        self.shape = (len(X), n_unique)

    def transform(self, X: np.ndarray):
        X = LabelEncoder().fit_transform(X)
        hot = np.zeros(self.shape, dtype='int8')
        for i, j in enumerate(X):
            hot[i][j - 1] = 1
        return hot
