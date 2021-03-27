import os
import numpy as np
from six.moves import cPickle

import cv2

b_color = np.array([0, 0, 0])
n = 16


def get_images():
    symbol_pack = {}
    for root, dirs, files in os.walk("/home/korpatiy/Рабочий стол/pyworks/ImageProcessing/processed_train"):
        if root[-1] not in symbol_pack:
            symbol_pack[root[-1]] = []
        for filename in files:
            symbol_pack[root[-1]].append(root + "/" + filename)
    return symbol_pack


def neighbour(x, y, dir_x, dir_y, value, image, result_image):
    for i in range(len(dir_y)):
        newX = (x + dir_x[i] + n) % n
        newY = (y + dir_y[i] + n) % n
        if np.array_equal(image[newX, newY], b_color):
            result_image[newX, newY] += value
    return result_image


def get_map(image):
    height, width, _ = image.shape
    result_image = np.zeros((n, n), np.float32)
    dirX = np.array([-1, 0, 1, 0])
    dirY = np.array([0, 1, 0, -1])
    dirX_1 = np.array([-1, -1, 1, 1])
    dirY_1 = np.array([-1, 1, 1, -1])
    for i in range(height):
        for j in range(width):
            if np.array_equal(image[i, j], b_color):
                result_image[i, j] += 1
                result_image = neighbour(i, j, dirX, dirY, 1 / 6, image, result_image)
                result_image = neighbour(i, j, dirX_1, dirY_1, 1 / 12, image, result_image)
    return result_image


def train():
    symbol_pack = get_images()
    model = {}
    for symbol, filenames in symbol_pack.items():
        for filename in filenames:
            image = cv2.imread(filename)
            if image is None:
                print("image " + filename + " not opened")
                continue
            map = get_map(image)
            if symbol not in model:
                model[symbol] = []
            model[symbol].append(map.ravel())
    with open('PModel.pickle', 'wb') as f:
        cPickle.dump(model, f)


train()
