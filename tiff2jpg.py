import matplotlib
import PIL
import pathlib
import matplotlib.pyplot as plt
import numpy as np
#import tensorflow as tf
import sys
from PIL import Image
import os
from os import listdir
import shutil

data_dir = sys.argv[1]
data_dir = pathlib.Path(data_dir)

image_count = len(list(data_dir.glob('*/*.tiff')))+len(list(data_dir.glob('*/*.jpg')))
print(image_count)

out_dir = "t2j_dir"

labels = {"advertisement": 0, "budget": 1, "email": 2, "file_folder": 3, "form": 4, "handwritten": 5, "invoice": 6, "letter": 7, "memo": 8, "news_article": 9, "pit37_v1": 10, "pozwolenie_uzytkowanie_obiektu_budowlanego": 11, "presentation": 12, "questionnaire": 13, "resume": 14, "scientific_publication": 15, "scientific_report": 16, "specification": 17, "umowa_na_odleglosc_odstapienie": 18, "umowa_o_dzielo": 19, "umowa_sprzedazy_samochodu": 20}

globs = []

i = 0
for label in labels:
	os.mkdir("t2j_dir" + "/" + label) 
	for images in os.listdir("train_set1" + "/" + label):
		if (images.endswith("jpg")):
			shutil.copy("train_set1" + "/" + label + "/" + images, out_dir + "/" + label )
		else:
			im = Image.open("train_set1" + "/" + label + "/" + images)
			images = images.replace(".tiff", "")
			im.save(out_dir + "/" + label + "/" + images + '.jpg')

#image_count2 = len(list(out_dir.glob('*/*.tiff')))+len(list(out_dir.glob('*/*.jpg')))
#print(image_count2)

#if image_count == image_count2:
#	print("All files converted to .jpg")
