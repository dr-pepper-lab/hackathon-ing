import pathlib
import shutil
import sys
import os
from PIL import Image

data_dir = pathlib.Path(sys.argv[1])

image_count = len(list(data_dir.glob('*/*.tiff')))+len(list(data_dir.glob('*/*.jpg')))
print(image_count)

out_dir = "prepared_source_images"
shutil.rmtree(out_dir, ignore_errors=True, onerror=None)
os.makedirs(out_dir)


labels = {"advertisement": 0, "budget": 1, "email": 2, "file_folder": 3, "form": 4, "handwritten": 5, "invoice": 6, "letter": 7, "memo": 8, "news_article": 9, "pit37_v1": 10, "pozwolenie_uzytkowanie_obiektu_budowlanego": 11, "presentation": 12, "questionnaire": 13, "resume": 14, "scientific_publication": 15, "scientific_report": 16, "specification": 17, "umowa_na_odleglosc_odstapienie": 18, "umowa_o_dzielo": 19, "umowa_sprzedazy_samochodu": 20}

globs = []

i = 0
for label in labels:
	os.mkdir(os.path.join(out_dir, label)) 
	for images in os.listdir(os.path.join(data_dir, label)):
		if (images.endswith("jpg")):
			shutil.copy(os.path.join(data_dir, label, images), os.path.join(out_dir, label))
		else:
			im = Image.open(os.path.join(data_dir, label, images))
			images = images.replace(".tiff", "")
			im.save(os.path.join(out_dir, label, images + '.jpg'))
