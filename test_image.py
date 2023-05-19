import numpy as np
import tensorflow as tf
import sys
import cv2
import os


def test_image(path, print_answer=False):

    img_height = 200
    img_width = 160

    model = tf.keras.models.load_model('IngModel.h5')

    img = tf.keras.utils.load_img(
        path, target_size=(img_height, img_width)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    class_names = ["advertisement", "budget", "email", "file_folder", "form", "handwritten", "invoice", "letter", "memo", "news_article", "pit37_v1", "pozwolenie_uzytkowanie_obiektu_budowlanego", "presentation", "questionnaire", "resume", "scientific_publication", "scientific_report", "specification", "umowa_na_odleglosc_odstapienie", "umowa_o_dzielo", "umowa_sprzedazy_samochodu"]
    if print_answer:
        print(
            "This image most likely belongs to {} with a {:.2f} percent confidence."
            .format(class_names[np.argmax(score)], 100 * np.max(score))
        )

        img = cv2.imread(path,cv2.IMREAD_COLOR)
        title = str(100*np.max(score))[0:2] + "% " +  class_names[np.argmax(score)]
        cv2.imshow(title, img)
        cv2.waitKey()
        cv2.destroyAllWindows()

    return(class_names[np.argmax(score)])

if __name__ == "__main__":

    pred_path = sys.argv[1]

    if os.path.isfile(pred_path):
        test_image(pred_path, True)
    elif os.path.isdir(pred_path):
        for img in os.scandir(pred_path):
            test_image(img.path, True)
