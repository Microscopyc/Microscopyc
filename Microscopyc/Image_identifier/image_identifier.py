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
from torchvision import transforms, models
from abc import ABC, abstractmethod

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


class BacteriaIdentifier:
    # Define the same Model Characteristics used during training:
    def __init__(self, num_strains=33, im_shape=(224, 224)):
        self.num_strains = num_strains
        self.model = self.load_model()
        self.transform = transforms.Compose(
            [
                transforms.Resize(im_shape, Image.LANCZOS),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def load_model(self):
        model = models.shufflenet_v2_x0_5(weights='IMAGENET1K_V1')
        model.fc = nn.Linear(in_features=1024, out_features=self.num_strains)
        return model

    def load_and_preprocess_usr_image(self, image_path):
        with Image.open(image_path).convert("RGB") as image:
            image = self.transform(image)
        return image.unsqueeze(0)  # Add a batch dimension

    def predict(self, image_path):
        image_tensor = self.load_and_preprocess_usr_image(image_path)
        with torch.no_grad():  # Disable gradient calculation
            outputs = self.model(image_tensor)
            _, predicted = torch.max(outputs, 1)
        return predicted.item()  # Return the predicted class index


class ImageProcessor:
    """
    Class responsible for loading and preprocessing the image.
    """

    def __init__(self, image_path):
        self.image_path = image_path
        self.image_cv = None
        self.binary_image = None
        self.contours = None

    def load_usr_image(self):
        self.image_cv = cv2.imread(self.image_path)

    def convert_to_grayscale(self):
        if self.image_cv is None:
            self.load_usr_image()
        return cv2.cvtColor(self.image_cv, cv2.COLOR_BGR2GRAY)

    def apply_threshold(self, gray_image):
        _, self.binary_image = cv2.threshold(
            gray_image, 127, 255, cv2.THRESH_BINARY_INV
        )

    def find_contours(self):
        self.contours, _ = cv2.findContours(
            self.binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )


class BacteriaInfo:
    def __init__(self, contours, binary_image):
        self.contours = contours
        self.binary_image = binary_image
        self.num_bacteria = len(contours)
        self.sizes = []
        self.shapes = []

    def sizes_and_shapes(self):
        for contour in self.contours:
            area = cv2.contourArea(contour)
            self.sizes.append(area)

            # Get bounding rectangle for shape analysis
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h if h != 0 else 0
            self.shapes.append(aspect_ratio)

    def mean_and_std(self):
        if not self.sizes:
            raise ValueError(
                "Bacteria sizes are empty. Cannot compute results."
            )
        return np.mean(self.sizes), np.std(self.sizes)

    def pixels_per_bacteria(self):
        if self.num_bacteria == 0:
            raise ZeroDivisionError(
                "No bacteria detected. Cannot compute pixels per bacteria."
            )
        return (self.binary_image.size // 255) / self.num_bacteria


class Shape(ABC):
    @abstractmethod
    def classify(self, aspect_ratio):
        pass


class Coccus(Shape):
    def classify(self, aspect_ratio):
        return "Coccus" if 0.9 <= aspect_ratio <= 1.1 else None


class Bacillus(Shape):
    def classify(self, aspect_ratio):
        return "Bacillus" if 0.2 <= aspect_ratio <= 0.8 else None


class Spiral(Shape):
    def classify(self, aspect_ratio):
        return "Spiral" if aspect_ratio < 0.2 else None


class BacteriaShapeClassifier:
    def __init__(self):
        self.shape_strategies = [Coccus(), Bacillus(), Spiral()]

    def classify_shape(self, aspect_ratio):
        for strategy in self.shape_strategies:
            result = strategy.classify(aspect_ratio)
            if result:
                return result
        return f"unknown, the ratio is {aspect_ratio}"


class ImageAnalysis:
    """Class responsible for managing the complete image analysis process."""

    def __init__(self, image_path):
        self.image_path = image_path
        self.image_processor = ImageProcessor(image_path)
        self.bacteria_info = None
        self.shape_classifier = BacteriaShapeClassifier()

    def process_image(self):
        self.image_processor.load_usr_image()
        gray_image = self.image_processor.convert_to_grayscale()
        self.image_processor.apply_threshold(gray_image)
        self.image_processor.find_contours()

        self.bacteria_info = BacteriaInfo(
            self.image_processor.contours, self.image_processor.binary_image
        )
        self.bacteria_info.sizes_and_shapes()
        try:
            mean_size, std_dev_size = self.bacteria_info.mean_and_std()
        except ValueError as e:
            print(f"Error calculating mean and std: {e}")
            mean_size, std_dev_size = 0, 0
        try:
            pixels_per_bacteria = self.bacteria_info.pixels_per_bacteria()
        except ZeroDivisionError as e:
            print(f"Error calculating pixels per bacteria: {e}")
            pixels_per_bacteria = 0

        shape_classes = [
            self.shape_classifier.classify_shape(ratio)
            for ratio in self.bacteria_info.shapes
        ]
        predominant_shape = (
            pd.Series(shape_classes).mode()[0] if shape_classes else "Unknown"
        )
        same_shape_percentage = (
            shape_classes.count(predominant_shape) / len(shape_classes) * 100
            if shape_classes
            else 0
        )

        # Create dictionary with results
        results = {
            "image": os.path.basename(self.image_path),
            "bacteria_per_pixel": pixels_per_bacteria,
            "mean_size": mean_size,
            "std_dev_size": std_dev_size,
            "groups": self.bacteria_info.num_bacteria,
            "same_shape_percentage": same_shape_percentage,
            "predominant_shape": predominant_shape,
        }

        return results


def analyze_images(image_dir):
    results_list = []

    # Process each image in the specified directory
    for filename in os.listdir(image_dir):
        if filename.endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(image_dir, filename)
            analysis = ImageAnalysis(image_path)
            results = analysis.process_image()
            results_list.append(results)

    # Create a DataFrame from the results list
    results_df = pd.DataFrame(results_list)

    return results_df


def save_report_to_textfile(sorted_results, filename):
    """Save the sorted results to a text file."""
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
    # Directory containing new user images
    new_images_dir = "/content/drive/MyDrive/Analyze"

    identifier = BacteriaIdentifier()

    # Iterate through all images in the user directory
    for image_name in os.listdir(new_images_dir):
        if image_name.endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(new_images_dir, image_name)
            predicted_class = identifier.predict(image_path)
            print(
                f"The predicted strain for {image_name} is: {predicted_class}")

    # Analyze all images in the user directory
    analysis_results = analyze_images(new_images_dir)

    # Save results to a CSV file
    analysis_results.to_csv("Images_analysis_results.csv", index=False)
    print("DataFrame saved")

    # Sort the DataFrame by bacteria_per_pixel
    sorted_results = analysis_results.sort_values(
        by="bacteria_per_pixel", ascending=False
    )

    # Save the report to a text file
    save_report_to_textfile(sorted_results, "Images_report.txt")


if __name__ == "__main__":
    main()
