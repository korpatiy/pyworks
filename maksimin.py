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
    classes = []
    data_set = read_file("iris.csv.gz")
    train_set, test_set = split_data(data_set)
    x1 = train_set[0]
    xf, T = get_first_distances(x1, train_set)
    classes.append(x1)
    classes.append(xf)

    min_distances = []
    flag = True

    while flag:
        for point in data_set:
            distances = []
            for cur_class in classes:
                distance = np.linalg.norm(point - cur_class)
                distances.append((point, distance))
            min_dist_id = distances.index(min(distances))
            min_distances.append(distances[min_dist_id])
        min_distances.sort(key=operator.itemgetter(1), reverse=True)
        if min_distances[0][1] > T:
            classes.append(min_distances[0][0])
        #max_dist = min_distances.index(max(min_distances))





main()
