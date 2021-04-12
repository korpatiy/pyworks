import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

np.random.seed(1)

train_images = []
train_labels = []
shape = (64, 64)
train_path = 'ImageProcessing/parsed_images'

for filename in os.listdir(train_path):
    for file in os.listdir(os.path.join(train_path, filename)):
        img = cv2.imread(os.path.join(train_path, filename, file))
        train_labels.append(filename)
        img = cv2.resize(img, shape)
        train_images.append(img)

# One Hot encoded
train_labels = pd.get_dummies(train_labels).values

train_images = np.array(train_images)

x_train, x_val, y_train, y_val = train_test_split(train_images, train_labels, random_state=1)

x_train = x_train.astype("float32") / 255
x_val = x_val.astype("float32") / 255

model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 64x64 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The third convolution
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The fourth convolution
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(36, activation='softmax')
])

model.compile(
    loss='categorical_crossentropy',
    metrics=['acc'],
    optimizer='adam'
)

model.summary()

history = model.fit(x_train, y_train, epochs=50, batch_size=50, validation_data = (x_val, y_val))

score = model.evaluate(x_val, y_val)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

model.save('model.h5')
del model