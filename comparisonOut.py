import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import roc_curve, confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC
from sklearn.utils import shuffle

np.seterr(divide='ignore', invalid='ignore')

def p_r_f1(labels, y_true, y_pred):
    tp = y_true == y_pred
    tp_bins = y_true[tp]
    tp_sum = np.bincount(tp_bins, minlength=len(labels))
    pred_sum = np.bincount(y_pred, minlength=len(labels))
    true_sum = np.bincount(y_true, minlength=len(labels))
    precision = tp_sum / pred_sum
    recall = tp_sum / true_sum
    f1 = 2 * (precision * recall) / (precision + recall)
    f1 = np.nan_to_num(f1)
    rows = zip(labels, precision, recall, f1)
    for row in rows:
        print(row)


def mse(y_true, y_pred):
    return np.average((y_true - y_pred) ** 2, axis=0)


def conf_matrix(s, y_true, y_pred):
    cm = np.zeros((s, s))
    for a, p in zip(y_true, y_pred):
        cm[a][p] += 1
    return cm


def r2(y_true, y_pred):
    correlation_matrix = np.corrcoef(y_true, y_pred)
    correlation_xy = correlation_matrix[0, 1]
    r_squared = correlation_xy ** 2
    return r_squared


def report(y_true, y_pred):
    labels = np.unique(y_pred)
    le = LabelEncoder()
    y_pred = le.fit_transform(y_pred)
    y_true = le.fit_transform(y_true)
    print("mse = " + repr(mse(y_true, y_pred)))
    p_r_f1(labels, y_true, y_pred)
    print("confusion_matrix:")
    print(conf_matrix(len(labels), y_true, y_pred))
    # print(confusion_matrix(y_true, y_pred))
    print("r-squared = " + repr(r2(y_true, y_pred)))


def chunks(lst, chunk_size):
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def cross_valid(x):
    x = shuffle(x)
    cross_x = chunks(x, 15)

    for idx in range(len(cross_x)):
        copy_x = cross_x.copy()
        x_test = cross_x[idx][:, :-1]
        y_test = cross_x[idx][:, -1]
        copy_x.pop(idx)
        x = np.concatenate(copy_x)
        x_train = x[:, :-1]
        y_train = x[:, -1]
        KNN_model = KNeighborsClassifier(n_neighbors=5)
        KNN_model.fit(x_train, y_train)
        KNN_predict = KNN_model.predict(x_test)
        print("matrix #" + repr(idx + 1))
        print(accuracy_score(y_test, KNN_predict))

def roc(y_true, y_pred):
    X = [1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1]
    Y = [1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1]
    Zx, Zy = zip(*[(x, y) for x, y in sorted(zip(X, Y))])
    score = np.array(Zx)
    y = np.array(Zy)
    fpr = []
    tpr = []
    P = sum(y)
    N = len(y) - P
    FP = 0
    TP = 0
    for i in range(len(score)):
        if y[i] == 1:
            TP = TP + 1
        if y[i] == 0:
            FP = FP + 1
        fpr.append(FP / float(N))
        tpr.append(TP / float(P))
    plt.plot(fpr, tpr)
    plt.title("ROC Curve")
    plt.xlabel("False positive rate")
    plt.ylabel("True Positive Rate")
    plt.show()

def main():
    iris = sns.load_dataset("iris")
    cross_valid(iris.values)
    x = iris.iloc[:, :-1].values
    y = iris["species"]
    x_train, x_test, y_train, y_test = \
        train_test_split(x, y, test_size=0.20, random_state=27)

    print("---------------------- KNN ----------------------")
    KNN_model = KNeighborsClassifier(n_neighbors=5)
    KNN_model.fit(x_train, y_train)
    KNN_predict = KNN_model.predict(x_test)
    report(y_test, KNN_predict)

    print("---------------------- KMeans ----------------------")
    KMeans_model = KMeans(n_clusters=3)
    KMeans_model.fit(x_train)
    KMeans_predict = KMeans_model.predict(x_test)
    report(y_test, KMeans_predict)

    print("---------------------- SMV ----------------------")
    SVM_model = LinearSVC(max_iter=1000)
    scaler = StandardScaler()
    x_test = scaler.fit_transform(x_test)
    x_train = scaler.fit_transform(x_train)
    SVM_model.fit(x_train, y_train)
    SVM_predict = SVM_model.predict(x_test)
    report(y_test, SVM_predict)


main()
