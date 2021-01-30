import math
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
    data = data.iloc[:, [0, 1, 2, 3]].values
    return data


def split_data(data):
    train, test = train_test_split(data, test_size=0.1)
    return train, test


def main():
    data = read_file("iris.csv.gz")
    train_set, test_set = split_data(data)
    k = 3
    tolerance = 0.0001
    max_iterations = 500
    centroids = {}

    for x in range(k):
        centroids[x] = data[x]

    for x in range(max_iterations):
        classes = {}
        for y in range(k):
            classes[y] = []
        # for centroid in centroids:
        for features in train_set:
            distances = [np.linalg.norm(features - centroids[centroid]) for centroid in centroids]
            cluster = distances.index(min(distances))
            classes[cluster].append(features)


    previous = dict(centroids)


    x = 56

main()
