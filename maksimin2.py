import math
import operator

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split


def euclidean_distance(vecA, vecB):
    distance = 0
    for x in range(len(vecB)):
        distance += pow((vecA[x]) - vecB[x], 2)
    return math.sqrt(distance)


def read_file(file):
    data = pd.read_csv(file)
    data["species"] = data["species"].map({"setosa": 0, "versicolor": 1, "virginica": 2})
    train_data = data.iloc[:, [0, 1, 2, 3]].values
    data = data.iloc[:, [0, 1, 2, 3, 4]].values
    return data, train_data


def split_data(data):
    train, test = train_test_split(data, test_size=0.1)
    return train, test


def get_first_distances(x1, train):
    max_dist = 0
    xf = []
    for x in range(len(train)):
        dist = euclidean_distance(x1, train[x])
        if dist > max_dist:
            max_dist = dist
            xf = train[x]
    return xf, max_dist / 2.0


def get_accuracy(test, predict):
    cnt = 0
    for x in range(len(test)):
        if test[x][-1] == predict[x]:
            cnt += 1
    return (cnt / float(len(test))) * 100.0


def main():
    classes = []
    data_set, train_set = read_file("iris.csv.gz")
    # train_set, test_set = split_data(data_set)
    x1 = train_set[0]
    xf, T = get_first_distances(x1, train_set)
    classes.append(x1)
    classes.append(xf)
    # distances = []
    flag = True
    while flag:
        distances = []
        for x in range(len(classes)):
            for y in range(len(train_set)):
                dist = euclidean_distance(classes[x], train_set[y])
                if x > 0:
                    cur_dist = distances[y][1]
                    if cur_dist > dist:
                        new_train = np.append(train_set[y], x)
                        distances[y] = (new_train, dist)
                else:
                    new_train = np.append(train_set[y], x)
                    distances.append((new_train, dist))

        distances.sort(key=operator.itemgetter(1), reverse=True)
        if (distances[0][1]) > T:
            classes.append(np.delete(distances[0][0], -1))
            # T = (T + distances[0][1]) / len(classes)
            # distances = []
        else:
            flag = False

    cnt0 = 0
    cnt1 = 0
    cnt2 = 0

    for x in range(len(distances)):
        b = distances[x][0][-1]
        if b == 0:
            cnt0 += 1
        if b == 1:
            cnt1 += 1
        if b == 2:
            cnt2 += 1

    print(cnt0)
    print(cnt1)
    print(cnt2)

    # accuracy = get_accuracy(data_set, distances)


main()
