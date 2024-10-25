"""
This module contain the form to classiffy differente type of bacteria in
images taken with an optical microscope, not used in training.
In this module, user images are also processed and compared with each other
with a table and final text report as output.
"""

# Imports here
import os
import torch
import torch.nn as nn
import numpy as np
import cv2
import pandas as pd

from PIL import Image
from torchvision import datasets, transforms, models

# Dataset used for training details
im_shape = (224, 224)
im_height, im_width = im_shape[0], im_shape[1]
num_strains = 32

transform = transforms.Compose(
    [
        transforms.Resize((224, 224), Image.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


# Functions:
# Define the same Model Characteristics used during training:
def load_model():
    model = models.shufflenet_v2_x0_5(pretrained=False)
    model.fc = nn.Linear(in_features=1024, out_features=num_strains)
    return model


def load_and_preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")  # ACA DUDA CON DEF PROCESS!!
    image = transform(image)
    return image.unsqueeze(0)  # Add a batch dimension


def predict(image_path):
    model = load_model()
    image_tensor = load_and_preprocess_image(image_path)

    with torch.no_grad():  # Disable gradient calculation
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)  # Get the index of highest score
    return predicted.item()  # Return the predicted class index


def process_image(image_path):
    """
    Process each image individually to extract information about classified
    bacteria and it's content.
    Returns a dictionary containing the analysis results.
    """
    # Load the image
    image_cv = cv2.imread(image_path)
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)

    # Apply a binary threshold to isolate bacteria
    _, binary_image = cv2.threshold(
        gray_image, 127, 255, cv2.THRESH_BINARY_INV)

    # Find contours of the bacteria
    contours, _ = cv2.findContours(
        binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Initialize variables for analysis
    num_bacteria = len(contours)
    sizes = []
    shapes = []

    # Iterate over contours to calculate sizes and area
    for contour in contours:
        area = cv2.contourArea(contour)
        sizes.append(area)

        # Get the bounding rectangle for the contour to calculate the shape
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h if h != 0 else 0  # Avoid division by zero
        shapes.append(aspect_ratio)

    # Calculate mean and standard deviation of the sizes
    mean_size = np.mean(sizes) if sizes else 0
    std_dev_size = np.std(sizes) if sizes else 0

    # Calculate the number of pixels in the binary image
    pixels_per_bacteria = (
        (binary_image.size // 255) / num_bacteria if num_bacteria > 0 else 0
    )

    # Calculate shape distribution
    shape_counts = pd.Series(shapes).value_counts()
    most_common_shape_count = (shape_counts.max() if
                               not shape_counts.empty else 0)
    predominant_shape_ratio = (shape_counts.idxmax() if
                               not shape_counts.empty else None)
    same_shape_percentage = (
        (most_common_shape_count / num_bacteria * 100) if
        num_bacteria > 0 else 0
    )
    different_shape_count = num_bacteria - most_common_shape_count

    # Determine if all shapes are similar (i.e., same aspect ratio)
    shape_uniformity = (
        len(shape_counts) == 1
    )  # True if all shapes are the same, False otherwise

    # Map aspect ratios to descriptive names
    if predominant_shape_ratio is not None:
        if predominant_shape_ratio == 1:
            predominant_shape = "spherical"
        elif predominant_shape_ratio < 1:
            predominant_shape = "flattened"
        else:
            predominant_shape = "elongated"
    else:
        predominant_shape = "unknown"

    # Create a results dictionary
    results = {
        "image": os.path.basename(image_path),
        "bacteria_per_pixel": pixels_per_bacteria,
        "mean_size": mean_size,
        "std_dev_size": std_dev_size,
        "groups": num_bacteria,
        # groups refers to the total number of distinct bacterial clusters
        # or individual bacteria detected in an image.
        "same_shape_percentage": same_shape_percentage,
        "different_shape_count": different_shape_count,
        "shape_uniformity": shape_uniformity,
        # "shape uniformity: false" indicates that not all the bacteria
        # detected have the same shape. This is determined by comparing
        # the aspect ratios of the bounding rectangles around each bacteria.
        "predominant_shape": predominant_shape,
    }

    return results


def analyze_images(image_dir):
    """
    Analyze all images in a specified directory and return a DataFrame of
    results.
    """
    results_list = []

    # Process each image in the specified directory
    for filename in os.listdir(image_dir):
        if filename.endswith((".png", ".jpg", ".jpeg")):
            # Adjust extensions as needed
            image_path = os.path.join(image_dir, filename)
            results = process_image(image_path)
            results_list.append(results)

    # Create a DataFrame from the results list
    results_df = pd.DataFrame(results_list)

    return results_df


def save_report_to_textfile(sorted_results, filename):
    """
    Save the sorted results to a text file.
    """
    with open(filename, "w") as f:
        f.write("Bacteria Count Report:\n")
        f.write("-------------------------------------------------\n")
        for index, row in sorted_results.iterrows():
            f.write(
                f"""Image: {row['image']}, Bacteria per Pixel:
                    {row['bacteria_per_pixel']:.4f}, Groups_of_bacteria:
                    {row['groups']}\n"""
            )
    print(f"Report saved to {filename}")


def main():
    # Directory containing images for training
    train_data_dir = "/path/to/training/data"
    # Directory containing new user images
    new_images_dir = "/path/to/user/images"

    total_set = datasets.ImageFolder(train_data_dir, transform)
    train_labels = ({value: key for (key, value)
                     in total_set.class_to_idx.items()})
    print(len(train_labels)), print(train_labels)
    # len(train_labels): gives the number of strains trained
    # train_labels: gives their scientific names

    # Iterate through all images in the user directory
    for image_name in os.listdir(new_images_dir):
        if image_name.endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(new_images_dir, image_name)
            predicted_class = predict(image_path)
            print(
                f"The predicted strain for {image_name} is: {predicted_class}")

    analysis_results = analyze_images(new_images_dir)

    # Save results to a CSV file
    analysis_results.to_csv("bacteria_analysis_results.csv", index=False)
    print("DataFrame saved")

    # Sort the DataFrame by bacteria_per_pixel
    sorted_results = analysis_results.sort_values(
        by="bacteria_per_pixel", ascending=False
    )

    # Save the report to a text file
    save_report_to_textfile(sorted_results, "bacteria_analysis_report.txt")


if __name__ == "__main__":
    main()
