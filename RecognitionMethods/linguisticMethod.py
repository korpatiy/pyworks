import os
import numpy as np

import cv2

b_color = np.array([0, 0, 0])
g_color = np.array([0, 255, 0])
y_color = np.array([255, 255, 0])
r_color = np.array([255, 0, 0])


def get_images():
    symbol_pack = {}
    for root, dirs, files in os.walk("/home/korpatiy/Рабочий стол/pyworks/ImageProcessing/processed_train"):
        if root[-1] not in symbol_pack:
            symbol_pack[root[-1]] = []
        for filename in files:
            symbol_pack[root[-1]].append(root + "/" + filename)
    return symbol_pack


def get_first_point(image):
    height, width, _ = image.shape
    x = -1
    y = -1
    for i in range(height):
        for j in range(width):
            if np.array_equal(image[i, j], b_color):
                x = i
                y = j
                break
            if x != -1:
                break
    return x, y


def neighbour(x, y, image):
    n = 16
    dirX = np.array([0, 1, 1, 1, 0, -1, -1, -1])
    dirY = np.array([1, 1, 0, -1, -1, -1, 0, 1])
    way = 0
    c_way = 1
    count = 0
    found = False
    new_x = 0
    new_y = 0
    for i in range(len(dirY)):
        newX = (x + dirX[i] + n) % n
        newY = (y + dirY[i] + n) % n
        if np.array_equal(image[newX, newY], b_color):
            count += 1
            if not found:
                way = c_way
                new_x = newX
                new_y = newY
                found = True
        else:
            c_way += 1
    return (new_x, new_y), way, count


def get_bypass_string(image):
    bypass_string = ""
    head = None
    curr_point = get_first_point(image)
    next_point = curr_point
    other_point = None

    point_list = []
    while True:
        curr_point = next_point
        if np.array_equal(image[curr_point], b_color):
            image[curr_point] = r_color
        next_point, way, count = neighbour(curr_point[0], curr_point[1], image)
        if way != 0:
            point_list.append(curr_point)
            if head is not None:
                bypass_string = bypass_string + "("
                next_point = head
                other_point, way, count = neighbour(next_point[0], next_point[1], image)
                if count == 1:
                    next_point = y_color
                    head = point_list[-1]
                if head is None:
                    head = point_list[-1]
                else:
                    point_list.append(curr_point)
                image[curr_point] = g_color
            else:
                head = point_list[-1]
                image[curr_point] = g_color
            bypass_string = bypass_string + "("
        if curr_point == next_point and head is not None:
            next_point = head
            bypass_string = bypass_string + ")"
            bypass_string = bypass_string + "("
        bypass_string = bypass_string + repr(way)
        if curr_point == next_point:
            break
    return 0


def train():
    symbol_pack = get_images()
    model = {}
    for symbol, filenames in symbol_pack.items():
        for filename in filenames:
            image = cv2.imread(filename)
            if image is None:
                print("image " + filename + " not opened")
                continue
            bypass_string = get_bypass_string(image, filename)
            if symbol not in model:
                model[symbol] = []
            model[symbol].append(bypass_string)


def main():
    train()


main()
