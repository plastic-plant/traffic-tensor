# Trains a Tensorflow classifier on a small set of traffic signs and saves the model.
#
# pip install matplotlib numpy pillow tensorflow keras layers Sequential
# python 1-train.py

# Output:
#
# Found 500 files belonging to 3 classes.
# Using 400 files for training.
# Found 500 files belonging to 3 classes.
# Using 100 files for validation.
# Found class names ['no-entry', 'no-park', 'no-stop']
#
# Model: "sequential"
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
# ┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
# ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
# │ rescaling_1 (Rescaling)              │ (None, 64, 64, 3)           │               0 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ conv2d (Conv2D)                      │ (None, 64, 64, 16)          │             448 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ max_pooling2d (MaxPooling2D)         │ (None, 32, 32, 16)          │               0 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ conv2d_1 (Conv2D)                    │ (None, 32, 32, 32)          │           4,640 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ max_pooling2d_1 (MaxPooling2D)       │ (None, 16, 16, 32)          │               0 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ conv2d_2 (Conv2D)                    │ (None, 16, 16, 64)          │          18,496 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ max_pooling2d_2 (MaxPooling2D)       │ (None, 8, 8, 64)            │               0 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ flatten (Flatten)                    │ (None, 4096)                │               0 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ dense (Dense)                        │ (None, 128)                 │         524,416 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ dense_1 (Dense)                      │ (None, 3)                   │             387 │
# └──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
#
#  Total params: 548,387 (2.09 MB)
#  Trainable params: 548,387 (2.09 MB)
#  Non-trainable params: 0 (0.00 B)
#
# Epoch  1/10  13/13 ━━━━━━━━━━━━━━━━━━━━ 1s 25ms/step - accuracy: 0.6712 - loss: 0.7602 - val_accuracy: 0.9900 - val_loss: 0.4177
# Epoch  2/10  13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step - accuracy: 0.9620 - loss: 0.3444 - val_accuracy: 1.0000 - val_loss: 0.0448
# Epoch  3/10  13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step - accuracy: 0.9659 - loss: 0.0791 - val_accuracy: 1.0000 - val_loss: 0.0233
# Epoch  4/10  13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 14ms/step - accuracy: 0.9964 - loss: 0.0144 - val_accuracy: 1.0000 - val_loss: 0.0044
# Epoch  5/10  13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step - accuracy: 0.9973 - loss: 0.0073 - val_accuracy: 1.0000 - val_loss: 0.0019
# Epoch 6/10   13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step - accuracy: 1.0000 - loss: 0.0081 - val_accuracy: 1.0000 - val_loss: 0.0010
# Epoch 7/10   13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 14ms/step - accuracy: 0.9984 - loss: 0.0049 - val_accuracy: 1.0000 - val_loss: 0.0015
# Epoch 8/10   13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 14ms/step - accuracy: 0.9911 - loss: 0.0190 - val_accuracy: 1.0000 - val_loss: 0.0039
# Epoch 9/10   13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 14ms/step - accuracy: 0.9992 - loss: 0.0033 - val_accuracy: 1.0000 - val_loss: 7.8286e-04
# Epoch 10/10  13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 14ms/step - accuracy: 1.0000 - loss: 0.0010 - val_accuracy: 1.0000 - val_loss: 0.0011
#
#
# Image no-entry.png most likely belongs to class no-entry with a 100.00% confidence.
# Image no-park-question.jpg most likely belongs to class no-park with a 91.43% confidence.
# Image no-park.png most likely belongs to class no-park with a 95.93% confidence.
# Image no-stop-question.jpg most likely belongs to class no-stop with a 99.97% confidence.
# Image no-stop.png most likely belongs to class no-stop with a 100.00% confidence.

import os
os.environ["KERAS_BACKEND"] = "tensorflow"  # @param ["tensorflow", "jax", "torch"]
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence | train_ds.take(1) | https://github.com/tensorflow/tensorflow/issues/62963
#
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed

import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import Sequential
import pathlib
import json


#-- Dataset
data_dir = pathlib.Path("images/train")
image_count = len(list(data_dir.glob('*/*.png')))

signs1 = list(data_dir.glob('no-entry/*'))
PIL.Image.open(str(signs1[0]))
PIL.Image.open(str(signs1[1]))

signs2 = list(data_dir.glob('no-park/*'))
PIL.Image.open(str(signs2[0]))
PIL.Image.open(str(signs2[1]))

signs3 = list(data_dir.glob('no-stop/*'))
PIL.Image.open(str(signs3[0]))
PIL.Image.open(str(signs3[1]))

### Dataset training and evaluation
batch_size = 32
img_height = 64
img_width = 64

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
print(f"Found class names {class_names}")

### Visualise
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

### Configure for performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

#### Dataset preperation
normalization_layer = layers.Rescaling(1./255)
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
print(np.min(first_image), np.max(first_image)) # Pixel values are now [0,1]


#-- Model
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

# Model compilation
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

### Visualise
model.summary()


# Train
epochs=10
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

### Visualise
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


#-- Predict
for filename in sorted(os.listdir("images/test/example")):
    image = tf.keras.utils.load_img(os.path.join("images/test/example", filename), target_size=(img_height, img_width))
    image_array = tf.keras.utils.img_to_array(image)
    image_array = tf.expand_dims(image_array, 0) # Create a batch

    predictions = model.predict(image_array)
    score = tf.nn.softmax(predictions[0])

    print(
        "Image {} most likely belongs to class {} with a {:.2f}% confidence."
        .format(filename, class_names[np.argmax(score)], 100 * np.max(score))
    )

# Save model and class mapping 
model.export('models/model.tf', format = "tf_saved_model")
model.save('models/model.keras')
with open('models/class_names.json', 'w') as file: json.dump(class_names, file)
