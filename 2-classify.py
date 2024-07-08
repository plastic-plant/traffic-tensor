# Classifies images with TensorFlow model and returns predicted labels.
#
# pip install matplotlib numpy pillow tensorflow keras layers Sequential tensorflow_datasets keras_cv
# python 2-classify.py

import json
import os
os.environ["KERAS_BACKEND"] = "tensorflow" 

import numpy as np
import tensorflow as tf
import numpy as np

# Model
model = tf.keras.models.load_model('models/model.keras')
class_names = json.load(open('models/class_names.json', 'r')) #  ['no-entry', 'no-park', 'no-stop']
batch_size = 32
img_height = 64
img_width = 64

# Predict examples
for filename in sorted(os.listdir("images/test/example")):
    image = tf.keras.utils.load_img(os.path.join("images/test/example", filename), target_size=(img_height, img_width))
    image_batch = tf.expand_dims(tf.keras.utils.img_to_array(image), 0)
    y_pred = model.predict(image_batch)
    best_prediction = y_pred[0]
    score = tf.nn.softmax(best_prediction)
    predicted_class = np.argmax(score)
    
    print(
        "Image {} most likely belongs to class {} with a {:.2f}% confidence."
        .format(filename, class_names[predicted_class], 100 * np.max(score))
    )

