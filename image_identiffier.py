"""Image_identiffier: this module contain the form to classiffy 
type of bacteria in images taken with an optical microscope, not 
used in training
"""

# Imports here
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import cv2

# import utils as ut

from PIL import Image
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)
from torchvision import datasets, transforms, models
from matplotlib import ticker

"""Iniciando el contenido de usuario"""

# Directory containing the images to analize:
train_data_dir = (
    "/content/drive/MyDrive/data"
)  # Directory containing images for training
new_images_dir = "/content/drive/MyDrive/Analyze"  # Directory containing new images

# Dataset details
im_shape = (224, 224)
im_height, im_width = im_shape[0], im_shape[1]
num_strains = 32

"""Define the same Model Characteristics used during training"""


def load_model():
    model = models.shufflenet_v2_x0_5(pretrained=False)
    model.fc = nn.Linear(in_features=1024, out_features=num_strains)
    return model


model = load_model()
model.load_state_dict(torch.load("bacteria_model.pth"))
model.eval()  # Set the model to evaluation mode

"""Load, preprocess and predict the class of new Images"""

transform = transforms.Compose(
    [
        transforms.Resize((224, 224), Image.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

total_set = datasets.ImageFolder(train_data_dir, transform)
train_labels = {value: key for (key, value) in total_set.class_to_idx.items()}

print(len(train_labels)), print(train_labels)


def load_and_preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    return image.unsqueeze(0)  # Add a batch dimension


def predict(image_path):
    image_tensor = load_and_preprocess_image(image_path)

    with torch.no_grad():  # Disable gradient calculation
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)  # Get the index of the highest score

    return predicted.item()  # Return the predicted class index


# Iterate through all images in the directory
for image_name in os.listdir(new_images_dir):
    if image_name.endswith((".png", ".jpg", ".jpeg")):  # Check for image files
        image_path = os.path.join(new_images_dir, image_name)
        predicted_class = predict(image_path)
        print(f"The predicted bacterial strain for {image_name} is: {predicted_class}")
