import numpy as np
from tqdm.auto import tqdm


class LinearRegression():
    def __init__(self, n_iter=10000):
        self.__params = None
        self.__intercept = None
        self.__n_iter = n_iter
        self.history = []
        self.lr = 0.01

    def predict(self, X):
        return np.dot(X, self.__params)

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        weights = np.zeros(X.shape[1])
        for _ in tqdm(range(self.__n_iter)):
            pred = X.dot(weights)
            self.history.append(self.__rmse(y, pred))
            grad = X.T @ (pred - y)
            weights -= self.lr * grad / len(X)
        self.__params = weights
        print("Complete")

    def get_params(self):
        return self.__params

    def __rmse(self, y_true, output):
        diff = np.power(y_true - output, 2)
        return np.sqrt(diff)


if __name__ == '__main__':
    import pandas as pd

    data  = pd.read_csv("data/insurance.csv")
    to_hot = ['sex', 'children', 'smoker', 'region']
    for col in data.columns:
        if col not in to_hot and col not in ['charges']:
            data[col] /= data[col].max()
    onehot = pd.get_dummies(data, columns=to_hot)
    onehot.charges = onehot.charges.apply(np.log)

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_squared_error

    X_train, X_test, y_train, y_test = train_test_split(
        data.drop("species", axis=1), data.species)
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
