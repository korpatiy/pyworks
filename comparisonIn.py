import pandas as pd
import seaborn as sns
import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, euclidean_distances, davies_bouldin_score
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC


def delta(ck, cl):
    values = np.ones([len(ck), len(cl)]) * 10000
    for i in range(0, len(ck)):
        for j in range(0, len(cl)):
            values[i, j] = np.linalg.norm(ck[i] - cl[j])

    return np.min(values)


def big_delta(ci):
    values = np.zeros([len(ci), len(ci)])
    for i in range(0, len(ci)):
        for j in range(0, len(ci)):
            values[i, j] = np.linalg.norm(ci[i] - ci[j])

    return np.max(values)


def dunn(k_list):
    print("dunn = ")
    deltas = np.ones([len(k_list), len(k_list)]) * 1000000
    big_deltas = np.zeros([len(k_list), 1])
    l_range = list(range(0, len(k_list)))

    for k in l_range:
        f = l_range[0:k] + l_range[k + 1:]
        for l in (l_range[0:k] + l_range[k + 1:]):
            deltas[k, l] = delta(k_list[k], k_list[l])
        big_deltas[k] = big_delta(k_list[k])

    di = np.min(deltas) / np.max(big_deltas)
    return di


def dbi(classes):
    print("dbi = ")
    ck = [np.mean(classes[k], axis=0) for k in range(len(classes))]
    s = [np.sum([np.linalg.norm(j - ck[l]) for j in classes[l]]) / len(classes[l]) for l in range(len(classes))]
    d = np.sum(max((s[k] + s[l]) / np.linalg.norm(ck[k] - ck[l]) for l in range(len(classes)) if l != k) for k in
               range(len(classes)))
    return 1 / 3 * d


def silhouette(classes):
    print("silhouette = ")
    s = 0
    for k in range(len(classes)):
        for i in classes[k]:
            a = np.sum([np.linalg.norm(i - j) for j in classes[k]]) / (len(classes[k]) - 1)
            b = min(
                [np.sum([np.linalg.norm(i - j) for j in classes[l]]) / len(classes[l]) for l in range(len(classes))
                 if l != k])
            s += (b - a) / max(a, b)
    return s / 150


def main():
    iris = sns.load_dataset("iris")
    x = iris.iloc[:, :-1].values
    y = iris["species"]
    le = LabelEncoder()
    labels = np.unique(y)
    labels = le.fit_transform(labels)
    print("---------------------- KNN ----------------------")
    KNN_model = KNeighborsClassifier(n_neighbors=5)
    KNN_model.fit(x, y)
    KNN_pred = KNN_model.predict(x)
    le = LabelEncoder()
    KNN_pred = le.fit_transform(KNN_pred)
    classes = [[x[j] for j in range(len(x)) if KNN_pred[j] == i] for i in range(3)]
    print(silhouette(classes))
    print(dbi(classes))
    print(dunn(classes))

    print("---------------------- KMeans ----------------------")
    KMeans_model = KMeans(n_clusters=3)
    KMeans_model.fit(x)
    KMeans_predict = KMeans_model.predict(x)
    classes = [[x[j] for j in range(len(x)) if KMeans_predict[j] == i] for i in range(3)]
    print(silhouette(classes))
    print(dbi(classes))
    print(dunn(classes))

    print("---------------------- SMV ----------------------")
    SVM_model = LinearSVC(max_iter=1000)
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    SVM_model.fit(x, y)
    SVM_predict = SVM_model.predict(x)
    le = LabelEncoder()
    SVM_predict = le.fit_transform(SVM_predict)
    classes = [[x[j] for j in range(len(x)) if SVM_predict[j] == i] for i in range(3)]
    print(silhouette(classes))
    print(dbi(classes))
    print(dunn(classes))


main()
