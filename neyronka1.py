import os

import cv2
import numpy as np
from tensorflow.python.keras.models import load_model

test_images = []
test_labels = []
shape = (64, 64)
test_path = 'ImageProcessing/Test'
for filename in os.listdir(test_path):
    for file in os.listdir(os.path.join(test_path, filename)):
        img = cv2.imread(os.path.join(test_path, filename, file))
        test_labels.append(filename)
        img = cv2.resize(img, shape)
        test_images.append(img)

test_images = np.array(test_images)

model = load_model('model.h5')

checkImage = test_images[0:1]
checklabel = test_labels[0:1]

output = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: 'A', 11: 'B', 12: 'C',
          13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J', 20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O',
          25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T', 30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z'}

predict = model.predict(np.array(checkImage))
X = np.array(checkImage)
print("Actual :- ", checklabel)
print("Predicted :- ", output[np.argmax(predict)])

for image in test_images:
    x = image
    x = np.array([x])
    predict = model.predict(np.array(x))
    print("Actual :- ", 'R')
    print("Predicted :- ", output[np.argmax(predict)])
