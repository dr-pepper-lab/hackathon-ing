from test_image import test_image
import os

path = "test_set"
img_height = 200
img_width = 160

labels = {"advertisement": 0, "budget": 1, "email": 2, "file_folder": 3, "form": 4, "handwritten": 5, "invoice": 6, "letter": 7, "memo": 8, "news_article": 9, "pit37_v1": 10, "pozwolenie_uzytkowanie_obiektu_budowlanego": 11, "presentation": 12, "questionnaire": 13, "resume": 14, "scientific_publication": 15, "scientific_report": 16, "specification": 17, "umowa_na_odleglosc_odstapienie": 18, "umowa_o_dzielo": 19, "umowa_sprzedazy_samochodu": 20}

for filename in os.listdir(path):
    f = os.path.join(path, filename)
    if os.path.isfile(f):
        with open("submission_file.csv", "a+") as fl:
            fl.write(filename + "," + str(labels[str(test_image(f))]) + "\n")
