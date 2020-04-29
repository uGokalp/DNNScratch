import numpy as np


class Knn:

    def __init__(self, k=2):
        self.k = k
        self.X = None
        self.y = None
        self.distances = None
        self.distance = self.euclidian

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X = np.array(X)
        self.y = np.array(y)

        assert self.k <= self.X.shape[
            1], "K can't be greater than number of features."

    def predict(self, X):
        X = np.array(X)
        return np.array([self.__predict(x) for x in X])

    def __predict(self, new_x):
        distances = np.array([self.distance(X, new_x) for X in self.X])

        idx = distances.argsort()
        k_idx = idx[:self.k]
        labels = [self.y[i] for i in k_idx]
        most_common = np.bincount(labels).argmax()
        return most_common

    def euclidian(self, x1, x2):
        """
        Computes the euclidian distance.
        """
        squared = np.square(x1 - x2).sum()
        return np.sqrt(squared)


if __name__ == '__main__':
    import pandas as pd

    data = pd.read_csv("data/scaled.csv")
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    X_train, X_test, y_train, y_test = train_test_split(
        data.drop("species", axis=1), data.species)
    knn = Knn(k=3)
    knn.fit(X_train, y_train)
    preds = knn.predict(X_test)
    print(accuracy_score(y_test, preds))
