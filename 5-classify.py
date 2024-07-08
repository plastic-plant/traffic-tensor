# Classifies images with Torch model and returns predicted labels.
#
# pip install PIL torch torchvision
# python 5-classify.py

# Output:
#
# Image negative-question.png most likely belongs to class option-a.
# Image no-entry.png most likely belongs to class no-entry.
# Image no-park-question.jpg most likely belongs to class no-park.
# Image no-park.png most likely belongs to class no-park.
# Image no-stop-question.jpg most likely belongs to class no-stop.
# Image no-stop.png most likely belongs to class no-stop.

import json
import os
import torch
from PIL import Image

import torchvision.transforms as transforms
import torch.nn as nn

model = torch.load('models/model.pth')
model.eval()
class_names = json.load(open('models/class_names.json', 'r')) #  ['no-entry', 'no-park', 'no-stop']

# Predict examples
for filename in sorted(os.listdir("images/test/example")):

    # Load and preprocess the image
    image = Image.open(os.path.join("images/test/example", filename)).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)

    # Classify the image
    with torch.no_grad():
        output = model(input_batch)

    # Get the predicted class label
    _, predicted_class = torch.max(output, 1)

    print(
        "Image {} most likely belongs to class {}."
        .format(filename, class_names[predicted_class.item()])
    )
