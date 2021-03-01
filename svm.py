import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pygments.lexers.csound import newline
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

    def fit(self, x, y, lr=0.001, epochs=500):
        self.x = x
        self.y = y
        self.n, self.d = x.shape
        self.w = np.random.randn(self.d)
        self.b = 0

        for i in range(epochs):
            margin = self.get_margin(x, y)
            misclassified_idx = np.where(margin < 1)[0]
            d_w = self.w - self.c * y[misclassified_idx].dot(x[misclassified_idx])
            self.w = self.w - lr * d_w
            d_b = - self.c * np.sum(y[misclassified_idx])
            self.b = self.b - lr * d_b
        self.support_vectors = np.where(self.get_margin(x, y) <= 1)[0]

    def plot(self):
        d = {-1: 'green', 1: 'red'}
        plt.scatter(self.x[:, 0], self.x[:, 1], c=[d[y] for y in self.y])

        ax = plt.gca()
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        xx = np.linspace(xlim[0], xlim[1], 30)
        yy = np.linspace(ylim[0], ylim[1], 30)
        YY, XX = np.meshgrid(yy, xx)
        xy = np.vstack([XX.ravel(), YY.ravel()]).T
        Z = self.decision_function(xy).reshape(XX.shape)

        ax.contour(XX, YY, Z, colors=['r', 'b', 'r'], levels=[-1, 0, 1], alpha=0.5,
                   linestyles=['--', '-', '--'], linewidths=[2.0, 2.0, 2.0])

        ax.scatter(self.x[:, 0][self.support_vectors], self.x[:, 1][self.support_vectors], s=100,
                   linewidth=1, facecolors='none', edgecolors='k')

        plt.show()


def load_data():
    iris = sns.load_dataset("iris")
    # iris = datasets.load_iris()
    iris = iris.tail(100)
    #iris = iris.head(100)
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(iris["species"])
    y[y == 0] = -1
    iris = iris.iloc[:, [2, 3]]
    return iris.values, y


def main():
    x, y = load_data()
    model = SVMSoftMargin(c=1000.0)
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    model.fit(x, y)
    model.plot()

main()
