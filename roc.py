import numpy
import matplotlib.pyplot

X = [1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1]
Y = [1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1]

Zx, Zy = zip(*[(x, y) for x, y in sorted(zip(X, Y))])

score = numpy.array(Zx)
y = numpy.array(Zy)
print(score)
print(y)

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
matplotlib.pyplot.plot(fpr, tpr)
matplotlib.pyplot.title("ROC Curve")
matplotlib.pyplot.xlabel("False positive rate")
matplotlib.pyplot.ylabel("True Positive Rate")
matplotlib.pyplot.show()
