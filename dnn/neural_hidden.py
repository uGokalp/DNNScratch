import numpy as np
from tqdm.auto import tqdm


class NeuralClassification:

    def __init__(self, hidden=20, lr=0.5, niter=2000):
        np.random.seed(42)
        self.hidden = hidden
        self.output = 1
        self.input_w = None
        self.hidden_w = None
        self.lr = lr
        self.iter = niter
        self.history = []
        self.nfeatures = None

    def train(self, X, y):
        X = np.array(X)
        self.nfeatures = X.size
        y = np.array(y)

        if not self.input_w:
            self.input_w = np.random.normal(scale=1 / np.sqrt(X.shape[1]),
                                            size=(X.shape[1], self.hidden))
            self.hidden_w = np.random.normal(scale=1 / np.sqrt(self.hidden),
                                             size=(self.hidden, self.output))

        for e in tqdm(range(self.iter)):

            for features, true in zip(X, y):
                self.grad_input = np.zeros(self.input_w.shape)
                self.grad_hidden = np.zeros(self.hidden_w.shape)
                w1, output = self.forward(features)
                self.backward(true, output, features)
                self.history.append(self.__cross_entropy(true, output))
                self.update_grad()

    def update_grad(self):
        self.hidden_w += self.lr * self.grad_hidden / self.nfeatures
        self.input_w += self.lr * self.grad_input / self.nfeatures

    def backward(self, true, output, X):
        output_error = (true - output) * output * (1 - output)
        hidden_error = np.dot(output_error, self.hidden_w.T)
        hidden_error_deriv = hidden_error * hidden_error * (1 - hidden_error)

        self.grad_input += hidden_error_deriv * X[:, None]
        self.grad_hidden += output_error * hidden_error_deriv[:, None]

    def forward(self, x):
        w1 = self.__sigmoid(x.dot(self.input_w))
        output = self.__sigmoid(w1.dot(self.hidden_w))
        return w1, output

    def predict(self, x):
        x = np.array(x.values.tolist())
        w1 = self.__sigmoid(x.dot(self.input_w))
        output = self.__sigmoid(w1.dot(self.hidden_w))
        return output

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
    pred = n.predict(X_test).round()
    print(classification_report(y_test, pred))
    pred = n.predict(X_test)
