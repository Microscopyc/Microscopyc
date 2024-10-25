"""This module contain the training model for classification of
bacteria in images taken with an optical microscope.
Script adapted from
@inproceedings{gallardo2020bacterialident,
title={Deep Learning for Fast Identification of Bacterial Strains in Resource
Constrained Devices},
author={Rafael Gallardo-García, Sofía Jarquín-Rodríguez, Beatriz
Beltrán-Martínez and Rodolfo Martínez},
booktitle={Aplicaciones Científicas y Tecnológicas de las Ciencias
Computacionales},
pages={67--78},
year={2020},
organization={BUAP} }
With part of DIBaS dataset:
@article{zielinski2017,
title={Deep learning approach to bacterial colony classification},
author={Zieli'nski, Bartosz and Plichta, Anna and Misztal, Krzysztof and
Spurek, Przemyslaw and Brzychczy-Wloch, Monika and Ocho'nska, Dorota},
journal={PloS one},
volume={12},
number={9},
pages={e0184554},
year={2017},
publisher={Public Library of Science San Francisco, CA USA} }
"""

# Imports here
import torch
import time
import numpy as np

from PIL import Image
from sklearn.model_selection import KFold
from torch import nn
from torch import optim
from torch.utils.data import SubsetRandomSampler
from torchvision import datasets, transforms, models


# Directory containing the images for training:
data_dir = "/path/to/training/data"

# Dataset details
dataset_version = "original"  # original or augmented
img_shape = (224, 224)
img_size = str(img_shape[0]) + "x" + str(img_shape[1])

train_batch_size = 33  # Batch size for training
val_test_batch_size = 33
feature_extract = False
pretrained = True
h_epochs = 10  # Number of training epochs
kfolds = 10  # 10

# Define transforms for input data
training_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224), Image.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

# TODO: Load the datasets with ImageFolder
total_set = datasets.ImageFolder(data_dir, transform=training_transforms)

# Defining folds
splits = KFold(n_splits=kfolds, shuffle=True, random_state=42)

train_labels = {value: key for (key, value) in total_set.class_to_idx.items()}


print(len(train_labels))
print(train_labels)


# Freeze pretrained model parameters to avoid backpropogating through them
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        print("Setting grad to false.")
        for param in model.parameters():
            param.requires_grad = False

    return model


def get_device():
    # Model and criterion to GPU
    if torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


models.shufflenet_v2_x0_5(pretrained=False)


def load_model():
    # Transfer Learning
    model = models.shufflenet_v2_x0_5(pretrained=pretrained)

    # Mode
    model = set_parameter_requires_grad(model, feature_extract)

    # Fine tuning
    # Build custom classifier
    model.fc = nn.Linear(in_features=1024, out_features=32)
    return model


def create_optimizer(model):
    # Parameters to update
    params_to_update = model.parameters()

    if feature_extract:
        params_to_update = []
        for param in model.parameters():
            if param.requires_grad is True:
                params_to_update.append(param)

    else:
        n_params = 0
        for param in model.parameters():
            if param.requires_grad is True:
                n_params += 1

    # Loss function and gradient descent

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(params_to_update, lr=0.001, weight_decay=0.000004)

    return criterion.to(get_device()), model.to(get_device()), optimizer


# Variables to store fold scores
train_acc = []
times = []

for fold, (train_idx, valid_idx) in enumerate(splits.split(total_set)):

    start_time = time.time()

    print("Fold : {}".format(fold))

    # Train and val samplers
    train_sampler = SubsetRandomSampler(train_idx)
    print("Samples in training:", len(train_sampler))
    valid_sampler = SubsetRandomSampler(valid_idx)
    print("Samples in test:", len(valid_sampler))

    # Train and val loaders
    train_loader = torch.utils.data.DataLoader(
        total_set, batch_size=train_batch_size, sampler=train_sampler
    )
    valid_loader = torch.utils.data.DataLoader(
        total_set, batch_size=1, sampler=valid_sampler
    )

    device = get_device()

    criterion, model, optimizer = create_optimizer(load_model())

    # Training
    for epoch in range(h_epochs):

        model.train()
        running_loss = 0.0
        running_corrects = 0
        trunning_corrects = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += (preds == labels).sum()
            trunning_corrects += preds.size(0)

        epoch_loss = running_loss / trunning_corrects
        epoch_acc = (running_corrects.double() * 100) / trunning_corrects
        train_acc.append(epoch_acc.item())

        print(
            "\t\t Training: Epoch({}) - Loss: {:.4f}, Acc: {:.4f}".format(
                epoch, epoch_loss, epoch_acc
            )
        )

        # Validation

        model.eval()

        vrunning_loss = 0.0
        vrunning_corrects = 0
        num_samples = 0

        for data, labels in valid_loader:

            data = data.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            with torch.no_grad():
                outputs = model(data)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

            vrunning_loss += loss.item() * data.size(0)
            vrunning_corrects += (preds == labels).sum()
            num_samples += preds.size(0)

        vepoch_loss = vrunning_loss / num_samples
        vepoch_acc = (vrunning_corrects.double() * 100) / num_samples

        print(
            "\t\t Validation({}) - Loss: {:.4f}, Acc: {:.4f}".format(
                epoch, vepoch_loss, vepoch_acc
            )
        )

    time_fold = time.time() - start_time
    times.append(time_fold)
    print("Total time per fold: %s seconds." % (time_fold))

torch.save(model.state_dict(), "bacteria_model.pth")  # Save the model weights
print("Model saved as bacterial_strain_model.pth")  # Confirmation of save

print("Train accuracy average: ", np.mean(train_acc) / 100)
print("Average time per fold (seconds):", np.mean(times))
