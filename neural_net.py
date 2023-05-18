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

data_dir = sys.argv[1]
data_dir = pathlib.Path(data_dir)

image_count = len(list(data_dir.glob('*/*.tiff')))+len(list(data_dir.glob('*/*.jpg')))
print(image_count)

labels = {"advertisement": 0, "budget": 1, "email": 2, "file_folder": 3, "form": 4, "handwritten": 5, "invoice": 6, "letter": 7, "memo": 8, "news_article": 9, "pit37_v1": 10, "pozwolenie_uzytkowanie_obiektu_budowlanego": 11, "presentation": 12, "questionnaire": 13, "resume": 14, "scientific_publication": 15, "scientific_report": 16, "specification": 17, "umowa_na_odleglosc_odstapienie": 18, "umowa_o_dzielo": 19, "umowa_sprzedazy_samochodu": 20}

globs = []

i = 0
for label in labels:
    globs.append(list(data_dir.glob(str(label) + '/*')))
    #print(globs)
    #PIL.Image.open(str(globs[i][0])).show()
    i += 1

batch_size = 32
img_height = 200
img_width = 160

train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_ds.class_names
print(class_names)

# visualize

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")

for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

normalization_layer = layers.Rescaling(1./255)

normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixel values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))

num_classes = len(class_names)

model = Sequential([
  layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

epochs=10
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

pred_path = str(data_dir) + '/advertisement/00133f89-017d-44f5-a549-e11cf5b5221a.tiff'

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
