import unittest
import cv2
import numpy as np
from Image_identifier.image_identifier import BacteriaInfo
from unittest.mock import MagicMock


class TestBacteriaInfoUnit(unittest.TestCase):
    def setUp(self):
        # Sample contours and binary image (mocked)
        self.contours = [MagicMock(), MagicMock()]
        self.binary_image = MagicMock()
        self.bacteria_info = BacteriaInfo(self.contours, self.binary_image)

    def test_sizes_and_shapes(self):
        cv2.contourArea = MagicMock(side_effect=[100, 200])
        cv2.boundingRect = MagicMock(
            side_effect=[(0, 0, 10, 20), (5, 5, 30, 40)]
            )
        self.bacteria_info.sizes_and_shapes()

        # Assert
        self.assertEqual(self.bacteria_info.sizes, [100, 200])
        self.assertEqual(self.bacteria_info.shapes, [0.5, 0.75])

    def test_mean_and_std(self):
        self.bacteria_info.sizes = [100, 200, 300]
        mean, std = self.bacteria_info.mean_and_std()

        # Assert
        self.assertEqual(mean, np.mean([100, 200, 300]))
        self.assertEqual(std, np.std([100, 200, 300]))

    def test_mean_and_std_empty_sizes(self):
        # Assert
        self.bacteria_info.sizes = []
        with self.assertRaises(ValueError):
            self.bacteria_info.mean_and_std()

    def test_pixels_per_bacteria(self):
        # Mock the binary image size
        self.binary_image.size = 1024
        self.bacteria_info.num_bacteria = 2
        result = self.bacteria_info.pixels_per_bacteria()

        # Assert
        self.assertEqual(result, (1024 // 255) / 2)

    def test_pixels_per_bacteria_zero_bacteria(self):
        # Assert
        self.bacteria_info.num_bacteria = 0
        with self.assertRaises(ZeroDivisionError):
            self.bacteria_info.pixels_per_bacteria()


if __name__ == "__main__":
    unittest.main()
