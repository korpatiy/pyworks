import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import operator
from sklearn.model_selection import train_test_split


# +
def euclidean_distance(vecA, vecB):
    distance = 0
    for x in range(len(vecB)):
        distance += pow((vecA[x]) - vecB[x], 2)
    return math.sqrt(distance)


# +
def manhattan_distance(vecA, vecB):
    distance = 0
    for x in range(len(vecB)):
        distance += abs((vecA[x]) - vecB[x])
    return distance


# is simply the proportion of disagreeing components
def hamming_distance(vecA, vecB):
    # intersect = list(set(vecA) & set(vecB))
    cnt = 0
    for x in range(len(vecA)):
        if vecA[x] != vecB[x]:
            cnt += 1
    return cnt / len(vecB)


# 1 -
def jaccard_distance(vecA, vecB):
    return 1 - np.minimum(vecA, vecB).sum() / np.maximum(vecA, vecB).sum()


# +
def cos_distance(vecA, vecB):
    def dotProduct(vecA, vecB):
        distance = 0
        for x in range(len(vecB)):
            distance += vecA[x] * vecB[x]
        return distance

    return 1 - dotProduct(vecA, vecB) / math.sqrt(dotProduct(vecA, vecA)) / math.sqrt(dotProduct(vecB, vecB))


def get_neighbors(train, test, k):
    distances = []
    for x in range(len(train)):
        dist = cos_distance(np.delete(test, -1), np.delete(train[x], -1))
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


def get_accuracy(test, predict):
    cnt = 0
    for x in range(len(test)):
        if test[x][-1] == predict[x]:
            cnt += 1
    return (cnt / float(len(test))) * 100.0


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
    predict_result = []
    for x in range(len(test_set)):
        neighbors = get_neighbors(train_set, test_set[x], 5)
        print(neighbors)
        result = get_classes(neighbors)
        predict_result.append(result)
        print('predicted class = ' + repr(result) + ',actual class = ' + repr(test_set[x][-1]))
    accuracy = get_accuracy(test_set, predict_result)
    print('accuracy: ' + repr(accuracy) + '%')


main()
