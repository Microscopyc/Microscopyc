import unittest
from unittest.mock import MagicMock
from Image_identifier.image_identifier import BacteriaInfo
import cv2


class TestBacteriaInfoMock(unittest.TestCase):

    def setUp(self):
        self.contours = [MagicMock(), MagicMock()]
        self.binary_image = MagicMock()
        self.bacteria_info = BacteriaInfo(self.contours, self.binary_image)

    def test_sizing_and_shapes_with_empty_contours(self):
        self.bacteria_info.contours = []
        self.bacteria_info.sizes_and_shapes()

        # Assert that sizes and shapes remain empty
        self.assertEqual(self.bacteria_info.sizes, [])
        self.assertEqual(self.bacteria_info.shapes, [])

    def test_mean_and_std_with_single_bacteria(self):
        self.bacteria_info.sizes = [100]
        mean, std = self.bacteria_info.mean_and_std()
        self.assertEqual(mean, 100)
        self.assertEqual(std, 0)

    def test_pixels_per_bacteria_with_no_bacteria(self):
        self.bacteria_info.num_bacteria = 0
        with self.assertRaises(ZeroDivisionError):
            self.bacteria_info.pixels_per_bacteria()

    # Simulate if boundingRect fails or returns unexpected values
    def test_sizes_and_shapes_with_invalid_contour(self):
        cv2.contourArea = MagicMock(side_effect=[100])
        # Invalid contour with zero height:
        cv2.boundingRect = MagicMock(side_effect=[(0, 0, 0, 0)])
        self.bacteria_info.sizes_and_shapes()

        # Assert
        self.assertEqual(self.bacteria_info.sizes, [100])
        # Aspect ratio should be 0 due to zero height
        self.assertEqual(self.bacteria_info.shapes, [0])


if __name__ == '__main__':
    unittest.main()
