import os
import sys

import numpy as np
import cv2
from six.moves import cPickle
from ImageProcessing.imageProcess import process_image
from RecognitionMethods.potentialMethod import get_map


def recognize(model, image, root_file):
    processed_image = process_image(image)
    new_file_name = root_file + "new" + ".png"
    cv2.imwrite(new_file_name, processed_image)
    image = cv2.imread(new_file_name)
    map = get_map(image).ravel()
    os.remove(new_file_name)
    distance = sys.maxsize
    symbol = None
    for cur_symbol, models in model.items():
        distances = [np.linalg.norm(map - models[model]) for model in range(len(models))]
        min_dist = min(distances)
        if min_dist < distance:
            distance = min_dist
            symbol = cur_symbol
    return symbol


def main():
    with open('/home/korpatiy/Рабочий стол/pyworks/RecognitionMethods/PModel.pickle', 'rb') as handle:
        model = cPickle.load(handle)
    for root, dirs, files in os.walk('/home/korpatiy/Рабочий стол/pyworks/RecognitionMethods/ImagesForRecognize'):
        for filename in files:
            root_file = root + "/" + filename
            image = cv2.imread(root_file)
            if image is None:
                print("image " + filename + " not opened")
                continue
            symbol = recognize(model, image, root_file)
            print("Original symbol is " + filename[0] + " recognized is " + symbol)



main()
