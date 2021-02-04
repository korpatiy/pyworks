import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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


def show_data(centroids, classes):
    colors = ["r", "g", "b"]
    for centroid in centroids:
        plt.scatter(centroids[centroid][0], centroids[centroid][1], s=100, marker="x")
    for classification in classes:
        color = colors[classification]
        for features in classes[classification]:
            plt.scatter(features[0], features[1], color=color, s=30)
    plt.show()


def main():
    data = read_file("iris.csv.gz")
    train_set, test_set = split_data(data)
    k = 3
    tolerance = 0.0001
    max_iterations = 500
    centroids = {}

    for x in range(k):
        centroids[x] = data[x]

    isOptimal = True

    for x in range(max_iterations):
        classes = {}
        for y in range(k):
            classes[y] = []
        # for centroid in centroids:
        for features in data:
            distances = [np.linalg.norm(features - centroids[centroid]) for centroid in centroids]
            cluster = distances.index(min(distances))
            classes[cluster].append(features)
        previous = dict(centroids)
        for cur_class in classes:
            centroids[cur_class] = np.average(classes[cur_class], axis=0)
        for centroid in centroids:
            original_centroid = previous[centroid]
            curr = centroids[centroid]
            if np.sum((curr - original_centroid) / original_centroid * 100.0) > tolerance:
                isOptimal = False
            if isOptimal:
                break

    show_data(centroids, classes)


main()
