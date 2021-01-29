from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import operator


def euclidean_distance(vecA, vecB):
    distance = 0
    for x in range(len(vecB)):
        distance += pow((vecA[x]) - vecB[x], 2)
    return math.sqrt(distance)


def read_file(file):
    data = pd.read_csv(file)
    data["species"] = data["species"].map({"setosa": 0, "versicolor": 1, "virginica": 2})
    data = data.iloc[:, [0, 1, 2, 3]].values
    return data


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


def get_classes(train, class_elem, k, distances):
    for x in range(len(train)):
        dist = euclidean_distance(class_elem, train[x])
        new_train = np.append(train[x], k)
        distances.append((new_train, dist))
    return distances


# distances.append((train[x], dist))


def main():
    classes = {}
    data_set = read_file("iris.csv.gz")
    train_set, test_set = split_data(data_set)
    x1 = train_set[0]
    xf, T = get_first_distances(x1, train_set)
    classes[0] = (x1, 0.0)
    classes[1] = (xf, 0.0)
    # classes.append(x1)
    # classes.append(xf)
    distances = []
    min_distances = []
    flag = True

    for x in range(len(classes)):
        distances = []
        for y in range(len(train_set)):
            dist = euclidean_distance(classes[x][0], train_set[y])
            distances.append((train_set[y], dist))
        classes[x] = distances



    # distances.sort(key=operator.itemgetter(1), reverse=True)

    while flag:
        for x in range(len(classes)):
            for y in range(len(train_set)):
                dist = euclidean_distance(classes[x], train_set[y])
                classes[x] = (train_set[y], dist)

        distances.sort(key=operator.itemgetter(1), reverse=True)
        if (distances[0][1]) > T:
            classes.append(np.delete(distances[0][0], -1))
        else:
            flag = False

    if x > 0:
        cur_dist = distances[y][1]
        if cur_dist > dist:
            new_train = np.append(train_set[y], x)
            distances[y] = (new_train, dist)
    else:
        new_train = np.append(train_set[y], x)
        distances.append((new_train, dist))





    for x in range(len(classes)):
        for y in range(len(distances)):
            if distances[y][0][-1] == x:
                test = distances[y]
                if distances[y][1] > T:
                    classes.append(np.delete(distances[y][0], -1))
                    break

                # np.delete(distances[y][0], -1)




    cnt = 0
    for x in range(len(distances)):
        b = distances[x][0][-1]
        if distances[x][0][-1] == 0:
            cnt += 1
    print(cnt)
    # get_classes(train_set, classes[x], x, distances)


main()
