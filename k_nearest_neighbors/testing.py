import pandas as pd
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()
data = pd.DataFrame(cancer.data, columns=cancer.feature_names)
data["label"] = cancer.target

X_train, X_test, y_train, y_test = train_test_split(data.drop("label", axis=1),
                                                    data.label)

res = np.linalg.norm(X_train)
print(res[0])