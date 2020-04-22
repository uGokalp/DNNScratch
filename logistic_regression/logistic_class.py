import numpy as np
from tqdm.auto import tqdm


class LogisticRegression:

    def __init__(self, n_iter=10000):
        self.__params = None
        self.__intercept = None
        self.__n_iter = n_iter
        self.history = []
        self.lr = 0.01

    def predict(self, X):
        return self.predict_proba(X).round()

    def predict_proba(self, X):
        return self.__sigmoid(np.dot(X, self.__params))

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        weights = np.zeros(X.shape[1])
        for _ in tqdm(range(self.__n_iter)):
            pred = self.__sigmoid(X @ weights)
            self.history.append(self.__cross_entropy(y, pred))
            grad = X.T @ (pred - y)
            weights -= self.lr * grad / len(X)
        self.__params = weights
        print("Complete")

    def get_params(self):
        return self.__params

    def __sigmoid(self, z):
        return np.divide(1, np.add(1, np.exp(-z)))

    def __cross_entropy(self, y_true, output):
        partA = np.multiply(-y_true,np.log(output))
        partB = np.multiply((1 - y_true), np.log(1 - output))
        return np.subtract(partA, partB).mean()


if __name__ == '__main__':
    import pandas as pd

    data = pd.read_csv("data/scaled.csv")
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        data.drop("species", axis=1), data.species)
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
