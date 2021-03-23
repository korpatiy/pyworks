import os
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
import cv2


def get_images():
    symbol_pack = {}
    for root, dirs, files in os.walk("train"):
        if root[-1] not in symbol_pack:
            symbol_pack[root[-1]] = []
        for filename in files:
            symbol_pack[root[-1]].append(root + "/" + filename)
    return symbol_pack


def process_image(image, filename):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.GaussianBlur(image, (5, 5), 0)
    if (image == 0).all():
        return None
    else:
        image = cv2.threshold(image, image.mean(), 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        if image[-1, -1] == 0:
            image = cv2.bitwise_not(image)

    return cv2.resize(image, (32, 32), cv2.INTER_NEAREST)


def work_process():
    symbol_pack = get_images()
    processed_images = {}
    for symbol, filenames in symbol_pack.items():
        for filename in filenames:
            image = cv2.imread(filename)
            if image is None:
                print("image " + filename + " not opened")
                continue
            new_image = process_image(image, filename)
            if symbol not in processed_images:
                processed_images[symbol] = []
            processed_images[symbol].append(new_image)
    new_dir = "processed_train"
    if not os.path.isdir(new_dir):
        os.mkdir(new_dir)
    os.chdir(new_dir)
    for symbol, images in processed_images.items():
        if not os.path.exists(symbol):
            os.makedirs(symbol)
        i = 0
        for image in images:
            if image is not None:
                cv2.imwrite(symbol + "/" + repr(i) + ".png", image)
            i += 1


def main():
    work_process()


main()
