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


def crop_image(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0, 0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, 0)
    bbox = diff.getbbox()
    left, upper, right, lower = bbox
    w, h = im.size
    bbox_new = (0, upper, w, lower)
    if bbox:
        return im.crop(bbox_new)
    else:
        print("error getbbox")


def black_white(image):
    fn = lambda x: 255 if x > 200 else 0
    image = image.convert('L').point(fn, mode='1')
    return image


def process_image(image):
    image = crop_image(image)
    image = image.resize((128, 128), Image.ANTIALIAS)
    image = black_white(image)
    return image


def work_process():
    symbol_pack = get_images()
    processed_images = {}
    for symbol, filenames in symbol_pack.items():
        for filename in filenames:
            image = Image.open(filename)
            # image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            if image is None:
                print("image " + filename + " not opened")
                continue
            new_image = process_image(image)
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
            image.save(symbol + "/" + repr(i) + ".png")
            # cv2.imwrite(symbol + "/" + repr(i) + ".png", image)
            i += 1


def main():
    work_process()


main()
