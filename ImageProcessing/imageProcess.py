import os

import cv2


def get_images():
    symbol_pack = {}
    for root, dirs, files in os.walk("train"):
        if root[-1] not in symbol_pack:
            symbol_pack[root[-1]] = []
        for filename in files:
            symbol_pack[root[-1]].append(root + "/" + filename)
    return symbol_pack


def contours(thresh):
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    c = max(cnts, key=cv2.contourArea)

    left = tuple(c[c[:, :, 0].argmin()][0])
    right = tuple(c[c[:, :, 0].argmax()][0])
    top = tuple(c[c[:, :, 1].argmin()][0])
    bottom = tuple(c[c[:, :, 1].argmax()][0])


def process_image(image, filename):
    blur = cv2.GaussianBlur(image, (3, 3), 0)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, gray.mean(), 255, cv2.THRESH_BINARY_INV)[1]
    x, y, w, h = cv2.boundingRect(thresh)
    cropped_image = thresh[y:y + h, x:x + w]
    if thresh[-1, -1] == 0:
        cropped_image = cv2.bitwise_not(cropped_image)
    resized_image = cv2.resize(cropped_image, (32, 32), cv2.INTER_NEAREST)
    return resized_image


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
