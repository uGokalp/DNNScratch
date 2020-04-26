"""Inspired from fast.ai"""


import numpy as np


class RandomForest:

    def __init__(self, X, y, n_trees, sample_size, min_leaf=3):
        np.random.seed(1)
        self.X = X
        self.y = y
        self.n_trees = n_trees
        self.sample_size = sample_size
        self.min_leaf = min_leaf
        self.trees = [self.create_tree() for n in range(n_trees)]

    def create_tree(self):
        idx = np.random.permutation(len(self.y))[:self.sample_size]
        return DecisionTree(X=self.X.iloc[idx], y=self.y[idx],
                            idxs=np.array(range(self.sample_size)),
                            min_leaf=self.min_leaf)

    def predict(self, X):
        return np.mean([tree.predict(X) for tree in self.trees], axis=0)


class DecisionTree:

    def __init__(self, X, y, idxs, min_leaf=5):
        np.random.seed(1)
        self.X = X
        self.y = y
        self.idxs = idxs
        self.value = np.mean(y[idxs])
        self.row, self.col = X.shape
        self.min_leaf = min_leaf
        self.score = float('inf')
        self.find_varsplit()

    def predict(self, x):
        return np.array([self.predict_row(xi) for xi in x])

    def predict_row(self, xi: np.ndarray):
        if self.is_leaf:
            return self.value
        if xi[self.var_idx] <= self.split:
            tree = self.lhs
        else:
            tree = self.rhs
        return tree.predict_row(xi)

    def find_varsplit(self):
        for i in range(self.col):
            self.find_better_split(i)
        if self.is_leaf:
            return
        x = self.split_col
        lhs = np.nonzero(x <= self.split)[0]  # Index
        rhs = np.nonzero(x > self.split)[0]
        self.lhs = DecisionTree(self.X, self.y, self.idxs[lhs])
        self.rhs = DecisionTree(self.X, self.y, self.idxs[rhs])

    def find_better_split(self, var_idx):
        """
        Idea: Want each group (branch) to have as low standard deviation as
        possible. This will try to split on every unique value of X. This is
        the same as minimizing the Root Mean Squared Error.

        Explanation: When I am talking about the standard deviation,
        I am talking about the standard deviation of Y, the dependent variable.

        :param var_idx: Index of the variable
        :return:
        """
        x = self.X.values[self.idxs, var_idx]
        y = self.y.values[self.idxs]

        sorted_idx = np.argsort(x)
        sorted_x, sorted_y = x[sorted_idx], y[sorted_idx]
        rhs_count, rhs_sum = self.row, sorted_y.sum()  # Actual rv
        rhs_sum2 = np.square(sorted_y).sum()  # Squared rv
        lhs_count, lhs_sum, lhs_sum2 = 0, 0., 0.

        for i in range(0, self.row - self.min_leaf - 1):
            xi, yi = sorted_x[i], sorted_y[i]
            lhs_count += 1
            rhs_count -= 1
            lhs_sum += yi
            rhs_sum -= yi
            square = np.square(yi)
            lhs_sum2 += square
            rhs_sum2 -= square

            if i < self.min_leaf or xi == sorted_x[i + 1]:
                continue  # if next x is the same x

            lhs_std = self.__std(lhs_count, lhs_sum2, lhs_sum)
            rhs_std = self.__std(rhs_count, rhs_sum2, rhs_sum)
            weighted_average = lhs_std * lhs_count + rhs_std * rhs_count
            if weighted_average < self.score:
                self.var_idx = var_idx  # Which variable
                self.score = weighted_average  # What score
                self.split = x[i]  # Where?

    def find_better_split_naive(self, var_idx):
        """
        Idea: Want each group (leaf) to have as low standard deviation as
        possible. This will try to split on every unique value of X. This is
        the same as minimizing the Root Mean Squared Error.

        O(n^2) algorithm

        Explanation: When I am talking about the standard deviation,
        I am talking about the standard deviation of Y, the dependent variable.

        :param var_idx: Index of the variable
        :return:
        """
        x = self.x.values[self.idxs, var_idx]
        y = self.y[self.idxs]

        for i in range(1, self.row - 1):
            lhs: np.bool = x <= x[i]
            rhs: np.bool = x > x[i]
            if rhs.sum() == 0:  # The terminal case when there is nothing to
                # the right of x
                continue
            lhs_std = y[lhs].std()
            rhs_std = y[rhs].std()
            weight_average: int = lhs_std * lhs.sum() + rhs_std * rhs.sum()
            if weight_average < self.score:
                self.var_idx = var_idx  # Which variable
                self.score = weight_average  # What score
                self.split = x[i]  # Where?

    @property
    def split_name(self):
        return self.X.columns[self.var_idx]

    @property
    def split_col(self):
        return self.X.values[self.idxs, self.var_idx]

    @property
    def is_leaf(self):
        return self.score == float('inf')

    def __std(self, count, x_squared, x):
        # We have Var(x) = E[x^2] - E[x]^2
        return np.sqrt((x_squared / count) - np.square(x / count))

    def __repr__(self):
        s = f'n: {self.n}; val:{self.value}'
        if not self.is_leaf:
            s += f'; score:{self.score}; split:{self.split}; var:' \
                 f'{self.split_name}'
        return s


if __name__ == '__main__':
    import pandas as pd

    data = pd.read_csv("data/insurance.csv")
    to_hot = ['sex', 'children', 'smoker', 'region']
    for col in data.columns:
        if col not in to_hot and col not in ['charges']:
            data[col] /= data[col].max()
    onehot = pd.get_dummies(data, columns=to_hot)
    onehot.charges = onehot.charges.apply(np.log)

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_squared_error

    X_train, X_test, y_train, y_test = train_test_split(
        onehot.drop("charges", axis=1), onehot.charges)
    clf = RandomForest(X_train, y_train, 100, len(X_train))
    preds = clf.predict(X_test.values)
    print("Preds shape", preds.shape)
    print(mean_squared_error(y_test, preds))
    print(r2_score(y_test, preds))
