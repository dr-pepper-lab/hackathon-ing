import numpy as np
import tensorflow as tf
import sys
import os

class_names = ["advertisement", "budget", "email", "file_folder", "form", "handwritten", "invoice", "letter", "memo", "news_article", "pit37_v1", "pozwolenie_uzytkowanie_obiektu_budowlanego", "presentation", "questionnaire", "resume", "scientific_publication", "scientific_report", "specification", "umowa_na_odleglosc_odstapienie", "umowa_o_dzielo", "umowa_sprzedazy_samochodu"]

img_height = 200
img_width = 160
model = tf.keras.models.load_model('IngModel.h5')

pred_path = sys.argv[1]

def test_image(path):
    img = tf.keras.utils.load_img(
        path, target_size=(img_height, img_width)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    img_class = class_names[np.argmax(score)]

if os.path.isdir(pred_path):
    for img in os.scandir(pred_path):
        test_image(img.path)