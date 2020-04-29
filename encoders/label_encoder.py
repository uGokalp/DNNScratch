from encoder import Encoder
import numpy as np


class LabelEncoder(Encoder):

    def __init__(self):
        self.mapper = None
        self.unique = None

    def fit(self, X: np.ndarray):
        self.unique = np.unique(X)
        n_unique = len(self.unique)
        self.mapper = {k: v for k, v in zip(self.unique, range(n_unique))}
        return None

    def transform(self, X: np.ndarray):
        accumulate = []
        for x in X:
            accumulate.append(self.mapper[x])
        return np.array(accumulate)
