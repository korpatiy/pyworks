import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets, preprocessing
from sklearn.preprocessing import StandardScaler


class SVMSoftMargin:
    def __init__(self, c=1.0):
        self.c = c
        self.w = None
        self.b = None
        self.x = None
        self.y = None
        self.n = 0
        self.d = 0
        self.support_vectors = None

    # (wTx-b)
    def decision_function(self, x):
        return x.dot(self.w) + self.b

    # M - Отступ y(wTx-b)
    def get_margin(self, x, y):
        return y * self.decision_function(x)

    # Функция потерь
    def cost(self, margin):
        return self.c * np.sum(np.maximum(0, 1 - margin)) + self.w.dot(self.w) / 2.0

    def fit(self, x, y, lr=0.001, epochs=100):
        self.x = x
        self.y = y
        self.n, self.d = x.shape
        self.w = np.random.randn(self.d)
        self.b = 0

        loss_array = []
        for i in range(epochs):
            margin = self.get_margin(x, y)
            loss = self.cost(margin)
            loss_array.append(loss)
            misclassified_pts_idx = np.where(margin < 1)[0]
            d_w = self.w - self.c * y[misclassified_pts_idx].dot(x[misclassified_pts_idx])
            self.w = self.w - lr * d_w
            d_b = - self.c * np.sum(y[misclassified_pts_idx])
            self.b = self.b - lr * d_b
        self.support_vectors = np.where(self.get_margin((x, y)) <= 1)[0]


def load_data():
    iris = sns.load_dataset("iris")
    # iris = datasets.load_iris()
    iris = iris.tail(100)
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(iris["species"])
    y[y == 0] = -1
    iris = iris.iloc[:, [2, 3]]
    return iris.values, y


if __name__ == '__main__':
    x, y = load_data()
    model = SVMSoftMargin(c=15.0)
    model.fit(x, y)

    p = 3
