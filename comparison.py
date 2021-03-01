import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix, classification_report, confusion_matrix, roc_curve, \
    plot_roc_curve, r2_score, davies_bouldin_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score


def p_r_f1(labels, y_true, y_pred):
    tp = y_true == y_pred
    tp_bins = y_true[tp]
    tp_sum = np.bincount(tp_bins, minlength=len(labels))
    pred_sum = np.bincount(y_pred, minlength=len(labels))
    true_sum = np.bincount(y_true, minlength=len(labels))
    precision = tp_sum / pred_sum
    recall = tp_sum / true_sum
    # plot_roc_curve(true_sum, pred_sum)
    f1 = 2 * (precision * recall) / (precision + recall)
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


def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()


def r2(y_true, y_pred):
    correlation_matrix = np.corrcoef(y_true, y_pred)
    correlation_xy = correlation_matrix[0, 1]
    r_squared = correlation_xy ** 2
    return r_squared


def dunn_idx(y_true, y_pred):
    mind = np.minimum(np.abs(y_true-y_pred))
    maxd = np.maximum(np.abs(y_true-y_pred))
    return mind / maxd


def report(y_true, y_pred):
    labels = np.unique(y_pred)
    le = LabelEncoder()
    y_pred = le.fit_transform(y_pred)
    y_true = le.fit_transform(y_true)
    print("mse = " + repr(mse(y_true, y_pred)))
    p_r_f1(labels, y_true, y_pred)
    cm = conf_matrix(len(labels), y_true, y_pred)
    # fpr, tpr = roc_curve(y_true, y_pred)
    print(r2_score(y_true, y_pred))
    print(r2(y_true, y_pred))
    davies_bouldin_score


if __name__ == '__main__':
    iris = sns.load_dataset("iris")
    x = iris.iloc[:, :-1].values
    y = iris["species"]
    x_train, x_test, y_train, y_test = \
        train_test_split(x, y, test_size=0.20, random_state=27)
    KNN_model = KNeighborsClassifier(n_neighbors=5)
    KNN_model.fit(x_train, y_train)
    KNN_predict = KNN_model.predict(x_test)
    report(y_test, KNN_predict)
    # print(accuracy_score(KNN_predict, y_test))
    # print(multilabel_confusion_matrix(KNN_predict, y_test))
    # print(confusion_matrix(y_test, KNN_predict))
    # print(classification_report(y_test, KNN_predict))
