import numpy as np
import tensorflow as tf
import sys
import cv2
import csv

def open_text(path):
    with open(path) as f:
        text = f.read()
        oc = get_occurences(text) + [len(text.split())]
    return(oc)

def get_occurences(text):
    dicts = [[('the', 940), ('and', 620), ('The', 248), ('for', 240), ('with', 240), ('you', 228), ('And', 220), ('that', 190), ('your', 172), ('Smoking', 166)],
    [('and', 208), ('the', 146), ('TOTAL', 140), ('TOBACCO', 135), ('QUARTER', 133), ('Total', 129), ('THE', 108), ('DATE', 102), ('for', 95), ('INC', 79)],
    [('the', 1301), ('and', 789), ('for', 445), ('that', 344), ('Subject:', 300), ('you', 242), ('with', 240), ('From:', 215), ('have', 208), ('will', 207)],
    [('END', 11), ('FILE', 9), ('111', 7), ('FOLDER', 6), ('MORRIS', 4), ('CTR', 4), ('ZOXIV', 4), ('PRODUCED', 4), ('EROM', 4), ('BEN', 4)],
    [('the', 927), ('and', 531), ('for', 299), ('this', 206), ('The', 178), ('DATE', 150), ('that', 138), ('Date', 133), ('AND', 129), ('any', 129)],
    [('the', 110), ('and', 78), ('you', 56), ('your', 56), ('for', 56), ('our', 35), ('You', 34), ('The', 29), ('how', 26), ('have', 26)],
    [('DATE', 228), ('INVOICE', 180), ('and', 155), ('TOTAL', 129), ('the', 115), ('DUE', 90), ('FOR', 82), ('AMOUNT', 81), ('Box', 76), ('for', 75)],
    [('the', 2863), ('and', 1605), ('for', 984), ('that', 579), ('The', 463), ('will', 387), ('with', 372), ('you', 360), ('are', 359), ('have', 336)],
    [('the', 2848), ('and', 1688), ('for', 887), ('The', 558), ('are', 411), ('that', 390), ('with', 385), ('will', 317), ('have', 281), ('this', 270)],
    [('the', 3679), ('and', 1742), ('that', 986), ('The', 665), ('for', 586), ('thc', 445), ('smoking', 445), ('with', 360), ('have', 339), ('tobacco', 333)],
    [('DANE', 990), ('roku', 905), ('poz', 667), ('nie', 656), ('dnia', 582), ('27-35', 548), ('mowa', 524), ('Urzad', 506), ('art', 503), ('podatkowym', 475)],
    [('pozwolenia', 636), ('1994', 520), ('dnia', 467), ('ilosci', 414), ('obiektu', 404), ('calkowitej', 399), ('podstawie', 365), ('udzielenie', 360), ('Nadzoru', 259), ('budowlanego', 257)],
    [('the', 2792), ('and', 1511), ('that', 699), ('for', 504), ('The', 457), ('tobacco', 413), ('Philip', 313), ('are', 299), ('with', 289), ('Morris', 277)],
    [('the', 1737), ('you', 870), ('and', 688), ('You', 419), ('your', 384), ('for', 323), ('that', 313), ('are', 257), ('not', 255), ('have', 254)],
    [('and', 4066), ('the', 2029), ('University', 1535), ('Research', 940), ('for', 865), ('Medical', 630), ('Professor', 542), ('Assistant', 518), ('with', 514), ('Department', 508)],
    [('the', 5641), ('and', 4485), ('The', 1341), ('thc', 1215), ('with', 1163), ('that', 1137), ('for', 1036), ('cells', 573), ('from', 540), ('DNA', 500)],
    [('the', 1902), ('and', 1285), ('for', 446), ('The', 413), ('with', 284), ('that', 253), ('were', 203), ('are', 200), ('che', 192), ('been', 165)],
    [('and', 291), ('AND', 282), ('FILTER', 257), ('DATE', 230), ('ADHESIVE', 197), ('Rod', 181), ('RODS', 174), ('PAPER', 169), ('Code', 168), ('SECTION', 167)],
    [('2000', 323), ('ochronie', 310), ('podstawie', 308), ('praw', 305), ('dnia', 304), ('odstapieniu', 293), ('przez', 278), ('produkt', 269), ('numer', 261), ('zawartej', 253)],
    [('Wykonawca', 1190), ('się', 942), ('Wykonawcy', 931), ('dziela', 801), ('Zamawiajacego', 705), ('dniu', 523), ('umowy', 447), ('tresci', 439), ('dzieła', 436), ('wobec', 378)],
    [('umowy', 905), ('jest', 862), ('nie', 703), ('dokumentu', 603), ('przedmiotu', 602), ('pojazd', 592), ('Sprzedajacy', 520), ('przedmiotem', 505), ('stanowi', 489), ('przez', 452)]]

    occurences = 21 * [0]
    dicts2 = []
    dicts2_inner = []

    for d in dicts:
        for el in d:
            dicts2_inner.append(el[0])
        dicts2.append(dicts2_inner)
        dicts2_inner = []

    labels = ["advertisement", "budget", "email", "file_folder", "form", "handwritten", "invoice", "letter", "memo", "news_article", "pit37_v1", "pozwolenie_uzytkowanie_obiektu_budowlanego", "presentation", "questionnaire", "resume", "scientific_publication", "scientific_report", "specification", "umowa_na_odleglosc_odstapienie", "umowa_o_dzielo", "umowa_sprzedazy_samochodu"]

    text_split = text.split()


    i=0
    for label in labels:
        for word in set(text_split):
            if word in dicts2[i]:
                occurences[i] += 1
        i += 1

    return(occurences)


from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

class_names = ["advertisement", "budget", "email", "file_folder", "form", "handwritten", "invoice", "letter", "memo", "news_article", "pit37_v1", "pozwolenie_uzytkowanie_obiektu_budowlanego", "presentation", "questionnaire", "resume", "scientific_publication", "scientific_report", "specification", "umowa_na_odleglosc_odstapienie", "umowa_o_dzielo", "umowa_sprzedazy_samochodu"]
labels2 = {"0": "advertisement", "1": "budget", "2": "email", "3": "file_folder", "4": "form", "5": "handwritten", "6": "invoice", "7": "letter", "8": "memo", "9": "news_article", "10": "pit37_v1", "11": "pozwolenie_uzytkowanie_obiektu_budowlanego", "12": "presentation", "13": "questionnaire", "14": "resume", "15": "scientific_publication", "16": "scientific_report", "17": "specification", "18": "umowa_na_odleglosc_odstapienie", "19": "umowa_o_dzielo", "20": "umowa_sprzedazy_samochodu"}

img_height = 200
img_width = 160

pred_path = sys.argv[1]
texts_path = sys.argv[2]

img = tf.keras.utils.load_img(
    pred_path, target_size=(img_height, img_width)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)


model = tf.keras.models.load_model('IngModel.h5')

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])


print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)

a1 = open_text(texts_path)
for j in predictions[0]:
    a1.append(j)

print(a1)

with open('list.csv', 'r') as file:
    reader = csv.reader(file)
    next(file)
    for row in reader:
        nr = ""
        #print(row)
        r0 = row[0]
        r0 = r0.replace(".tiff", "")
        fl = r0
        cat = labels2[str(row[1])]
        path = "prepared_source_texts/" + str(cat)+"/"
        path2 = "prepared_source_images/" + str(cat)+"/"
        pred_path = path2 + str(row[0])
        try:
            pred_path2 = pred_path + ".tiff"
            img = tf.keras.utils.load_img(
    	        pred_path, target_size=(img_height, img_width)
            )
        except:
            pred_path += ".jpg"
            img = tf.keras.utils.load_img(
                pred_path, target_size=(img_height, img_width)
            )

        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)


        predictions = model.predict(img_array)

        #print(path + fl + ".txt")
        ar = (open_text(path + fl + ".txt"))
        for j in predictions[0]:
             ar.append(j)
        vals = (','.join(map(str, ar)) )
        nr = row[0] + "," + vals + "," + row[1]
        print(nr)
        with open("vals.csv", "a+") as f:
            f.write(nr + "\n")
