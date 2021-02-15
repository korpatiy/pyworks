from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import operator

f = 5


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
    return xf, max_dist, max_dist / 2.0


def get_classes(train, class_elem, k, distances):
    for x in range(len(train)):
        dist = euclidean_distance(class_elem, train[x])
        new_train = np.append(train[x], k)
        distances.append((new_train, dist))
    return distances


def show_classes_base():
    iris = pd.read_csv("iris.csv.gz")
    for n in range(0, 150):
        if iris['species'][n] == 'setosa':
            plt.scatter(iris['sepal_length'][n], iris['sepal_width'][n], color='red')
            plt.xlabel('sepal_length')
            plt.ylabel('sepal_width')
        elif iris['species'][n] == 'versicolor':
            plt.scatter(iris['sepal_length'][n], iris['sepal_width'][n], color='blue')
            plt.xlabel('sepal_length')
            plt.ylabel('sepal_width')
        elif iris['species'][n] == 'virginica':
            plt.scatter(iris['sepal_length'][n], iris['sepal_width'][n], color='green')
            plt.xlabel('sepal_length')
            plt.ylabel('sepal_width')
    plt.show()


def show_new_classes(classes, classes_show):
    colors = ["r", "g", "b"]
    for curr_class in range(len(classes)):
        plt.scatter(classes[curr_class][0], classes[curr_class][1], s=100, marker="x")
    for point in classes_show:
        color = colors[point]
        for curr_point in classes_show[point]:
            plt.scatter(curr_point[0], curr_point[1], color=color, s=10)
    plt.show()


def main():
    classes = []
    data_set = read_file("iris.csv.gz")
    train_set, test_set = split_data(data_set)
    x1 = data_set[0]
    xf, prev_dist, threshold = get_first_distances(x1, data_set)
    classes.append(x1)
    classes.append(xf)
    flag = True

    while flag:
        classes_show = {}
        for y in range(len(classes)):
            classes_show[y] = []
        min_distances = []
        for point in data_set:
            distances = []
            for curr_class in classes:
                distance = np.linalg.norm(point - curr_class)
                distances.append((point, distance))
            min_dist_id = distances.index(min(distances))
            min_distances.append(distances[min_dist_id])
            classes_show[min_dist_id].append(point)

        min_distances.sort(key=operator.itemgetter(1), reverse=True)
        if min_distances[0][1] > threshold:
            classes.append(min_distances[0][0])
            class_distances_collect = []
            for curr_class in classes:
                class_distances = [np.linalg.norm(curr_class - classes[class_id]) for class_id in
                                   range(len(classes))]
                class_distances_collect.append(class_distances)
            new_threshold = np.sum(np.unique(class_distances_collect))
            #v = np.intersect1d(vecA, vecB)
            threshold = new_threshold / len(classes)
        else:
            flag = False

    print('Count of classes = ' + repr(len(classes)))
    show_classes_base()
    #show_new_classes(classes, classes_show)
    # max_dist = min_distances.index(max(min_distances))


main()
