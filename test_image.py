import matplotlib
import PIL
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import sys
import cv2

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

img = cv2.imread(sys.argv[1] ,cv2.IMREAD_COLOR)
img = cv2.putText(img, class_names[np.argmax(score)], (50, 180), cv2.QT_FONT_NORMAL, 
                   4, (194,24,91), 10, cv2.LINE_AA)
cv2.imshow('Tested image', img)
cv2.waitKey()
cv2.destroyAllWindows()