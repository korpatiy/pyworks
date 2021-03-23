import os
import numpy as np

import cv2

b_color = np.array([255, 255, 255])
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
    blur = cv2.GaussianBlur(image, (3, 3), 0)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, gray.mean(), 255, cv2.THRESH_BINARY_INV)[1]
    x, y, w, h = cv2.boundingRect(thresh)
    return x, y


def neighbour(x, y, image):
    dirX = np.array([0, 1, 1, 1, 0, -1, -1, -1])
    dirY = np.array([1, 1, 0, -1, -1, -1, 0, 1])
    way = 0
    c_way = 1
    count = 0
    found = False
    new_x = 0
    new_y = 0
    for i in range(len(dirY)):
        newX = (x + dirX[i] + 32) % 32
        newY = (y + dirY[i] + 32) % 32
        if np.array_equal(image[newX, newY], b_color):
            count += 1
            if not found:
                way = c_way
                new_x = x + newX
                new_y = y + newY
                found = True
        else:
            c_way += 1
    return (new_x, new_y), way, count


def get_bypass_string(image, filename):
    bypass_string = ""
    head = None
    curr_point = None
    next_point = get_first_point(image)
    other_point = None

    point_list = []
    while curr_point != next_point:
        curr_point = next_point
        if np.array_equal(image[curr_point], b_color):
            image[curr_point] = r_color
        next_point, way, count = neighbour(curr_point[0], curr_point[1], image)
        if way != 0:
            point_list.append(curr_point)
            if head is not None:
                bypass_string = bypass_string + "("
                next_point = point_list[-1]
                other_point, way, count = neighbour(next_point[0], next_point[1], image)
                if count == 1:
                    next_point = y_color
                    head = point_list[-1]
                if head is None:
                    head = curr_point
                else:
                    point_list.append(curr_point)
                    curr_point = g_color
            else:
                head = curr_point
                image[curr_point] = g_color
            bypass_string = bypass_string + "("
        if curr_point == next_point and head is not None:
            next_point = head
            bypass_string = bypass_string + ")"
            bypass_string = bypass_string + "("
        bypass_string = bypass_string + repr(way)
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
