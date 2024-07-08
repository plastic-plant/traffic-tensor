# Trains a Torch classifier on a small set of traffic signs and saves the model.
#
# pip install onnx
# python 4-train.py

# Output:
#
# Epoch  1/10, Loss: 0.7478
# Epoch  2/10, Loss: 0.1931
# Epoch  3/10, Loss: 0.0989
# Epoch  4/10, Loss: 0.0647
# Epoch  5/10, Loss: 0.0531
# Epoch  6/10, Loss: 0.0447
# Epoch  7/10, Loss: 0.0398
# Epoch  8/10, Loss: 0.0375
# Epoch  9/10, Loss: 0.0330
# Epoch 10/10, Loss: 0.0288

import pathlib
import torch
import torchvision
from torchvision.transforms import transforms

#-- Dataset
data_dir = pathlib.Path("images/train")
image_count = len(list(data_dir.glob('*/*.png')))

import torch.nn as nn
import torch.optim as optim

# Define the transformation for the input images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load the dataset
dataset = torchvision.datasets.ImageFolder(root=data_dir, transform=transform)

# Create the data loader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

# Define the model architecture
model = torchvision.models.resnet18(pretrained=True)
num_classes = len(dataset.classes)
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Train the model
num_epochs = 14
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):
    running_loss = 0.0
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(dataloader):.4f}")


# Save the model to disk and export to ONNX format
torch.save(model, 'models/model.pth')
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy_input, "models/model.onnx", input_names=["input"], output_names=["output"])
