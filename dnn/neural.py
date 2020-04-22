import numpy as np
from tqdm.auto import tqdm


class NeuralClassification:

    def __init__(self, lr=0.5, iter=2000):
        np.random.seed(42)
        self.weight = None
        self.lr = lr
        self.iter = iter

    def train(self, X, y):
        X = np.array(X)
        y = np.array(y)

        if not self.weight:
            self.weight = np.random.normal(scale=1 / np.sqrt(X.shape[1]),
                                           size=X.shape[1])
        for e in tqdm(range(self.iter)):
            grad_accumulated = np.zeros(self.weight.shape)
            for features, true in zip(X, y):
                output = self.__sigmoid(features.dot(self.weight))
                error = (true - output) * output * (1 - output)
                grad_accumulated += np.multiply(error, features)
            self.update_grad(X, grad_accumulated)

    def update_grad(self, X, grad_accumulated):
        self.weight += np.multiply(self.lr, grad_accumulated) / X.size

    def backward(self):
        pass

    def forward(self, x):
        return self.__sigmoid(x.dot(self.weight))

    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def __cross_entropy(self, y_true, output):
        partA = np.multiply(-y_true, np.log(output))
        partB = np.multiply((1 - y_true), np.log(1 - output))
        return np.subtract(partA, partB).mean()


if __name__ == '__main__':
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report

    data = pd.read_csv("data/scaled.csv", index_col=0);
    X_train, X_test, y_train, y_test = train_test_split(
        data.drop("species", axis=1), data.species)
    n = NeuralClassification()
    n.train(X_train, y_train)
    pred = n.forward(X_test).round()
    print(classification_report(y_test, pred))
