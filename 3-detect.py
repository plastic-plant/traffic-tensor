# Find circles with Hough circle transform in OpenCV and classifies cutouts as traffic signs or negatives.
#
# pip install tensorflow keras opencv-python
# python 2-classify.py

import numpy as np
import tensorflow as tf
import cv2
import json
import os
os.environ["KERAS_BACKEND"] = "tensorflow" 

model = tf.keras.models.load_model('models/model.keras')
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
class_names = json.load(open('models/class_names.json', 'r')) #  ['no-entry', 'no-park', 'no-stop']
img_height = 64
img_width = 64

results = []

for filename in sorted(os.listdir("images/test/question")):
    imagepath = os.path.join("images/test/question", filename)

    # Find circles with Hough Circle Transform in OpenCV
    image = cv.imread(imagepath, cv.IMREAD_COLOR)
    blurredGray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blurredGray = cv.medianBlur(blurredGray, 3)
    circles = cv.HoughCircles(blurredGray, cv.HOUGH_GRADIENT, 1, blurredGray.shape[0] / 8, param1=100, param2=30, minRadius=10, maxRadius=30)

    if circles is not None:
        print("Found %d circles", len(circles))
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            
            # Circle center
            center = (i[0], i[1])
            # cv.circle(image, center, 1, (0, 100, 100), 1)
            
            # Circle outline
            radius = i[2]
            # cv.circle(image, center, radius, (255, 0, 255), 2)

            # Circle bounding box
            x = center[0] - radius
            y = center[1] - radius
            w = h = 2 * radius
            # cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Copy bounding box from image
            cropped_cvimage = image[y:y+h, x:x+w]
            
            # Classify the detected circles with traffic sign classifier
            resized_image = cv2.resize(cropped_cvimage, (img_width, img_height))
            tfimage = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
            tensor = tf.convert_to_tensor(tfimage, dtype=tf.float32)
            image_batch = tf.expand_dims(tf.keras.utils.img_to_array(tensor), axis=0)
            
            # Label prediction
            predictions = probability_model.predict(image_batch)
            best_prediction = predictions[0]
            score = tf.nn.softmax(best_prediction)
            predicted_class = np.argmax(score)

            print(
                "Cropped image {} most likely belongs to class {} with a {:.2f}% confidence."
                .format(filename, class_names[predicted_class], 100 * np.max(score))
            )
            # cv2.imshow("cropped image", cropped_cvimage)
            # cv2.waitKey(0)

            results.append({
                'filename': filename,
                'x': int(x),
                'y': int(y),
                'label': class_names[predicted_class],
                'confidence': 100 * np.max(score)
            })

    # cv.imshow("detected circles", image)
    # cv.waitKey(0)

# Write results to a JSON file
with open('3-detect-results.json', 'w') as file:
    json.dump(results, file, indent = 4)
