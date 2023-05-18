import matplotlib
import PIL
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import sys

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

img_height = 200
img_width = 160

pred_path = sys.argv[1]

img = tf.keras.utils.load_img(
    pred_path, target_size=(img_height, img_width)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)