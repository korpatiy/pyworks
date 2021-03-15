import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import LinearSVC


def roc(y_true, y_pred):
    Zx, Zy = zip(*[(x, y) for x, y in sorted(zip(y_true, y_pred))])
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
    iris = iris.tail(100)
    x = iris.iloc[:, :-1].values
    y = iris["species"]
    SVM_model = LinearSVC(max_iter=1000)
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    SVM_model.fit(x, y)
    SVM_predict = SVM_model.predict(x)
    le = LabelEncoder()
    y = le.fit_transform(y)
    SVM_predict = le.fit_transform(SVM_predict)
    roc(y,SVM_predict)

main()