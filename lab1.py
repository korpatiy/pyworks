import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import operator
from sklearn.model_selection import train_test_split


def euclidean_distance(inst1, inst2, length):
    distance = 0
    for x in range(length):
        distance += pow((inst1[x]) - inst2[x], 2)
    return math.sqrt(distance)


def hamming_distance(inst1, inst2, length):
    distance = 0
    for x in range(length):
        distance += abs((inst1[x]) - inst2[x])
    return distance


def manhattan_distance(inst1, inst2, length):
    distance = 0
    for x in range(length):
        distance += abs()


def get_neighbors(train, test, k):
    distances = []
    length = len(test) - 1
    for x in range(len(train)):
        dist = euclidean_distance(test, train[x], length)
        distances.append((train[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors


def get_classes(neighbors):
    class_votes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in class_votes:
            class_votes[response] += 1
        else:
            class_votes[response] = 1
    sorted_votes = sorted(class_votes.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_votes[0][0]


def read_file(file):
    data = pd.read_csv(file)
    data["species"] = data["species"].map({"setosa": 0, "versicolor": 1, "virginica": 2})
    data = data.iloc[:, [0, 1, 2, 3, 4]].values
    return data


def split_data(data):
    train, test = train_test_split(data, test_size=0.1)
    return train, test


def main():
    data_set = read_file("iris.csv.gz")
    train_set, test_set = split_data(data_set)
    # print('Train set' + train_set)
    # print('Test set' + test_set)
    # print ('Train set len: ' + repr(len(train_set)))
    # print ('Test set len: ' + repr(len(test_set)))
    for x in range(len(test_set)):
        neighbors = get_neighbors(train_set, test_set[x], 3)
        print(neighbors)
        result = get_classes(neighbors)

main()
