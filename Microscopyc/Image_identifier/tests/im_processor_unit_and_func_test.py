import unittest
from unittest.mock import patch, MagicMock
import cv2
from Image_identifier.image_identifier import ImageProcessor


class TestImageProcessorUnit(unittest.TestCase):
    def setUp(self):
        self.image_path = "test_image.jpg"
        self.processor = ImageProcessor(self.image_path)

    @patch("cv2.imread")
    def test_load_usr_image(self, mock_imread):
        mock_image = MagicMock()
        mock_imread.return_value = mock_image
        self.processor.load_usr_image()

        # Assert
        mock_imread.assert_called_once_with(self.image_path)
        self.assertEqual(self.processor.image_cv, mock_image)

    @patch("cv2.imread")
    @patch("cv2.cvtColor")
    def test_convert_to_grayscale(self, mock_cvtColor, mock_imread):
        mock_image = MagicMock()
        mock_grayscale_image = MagicMock()
        mock_imread.return_value = mock_image
        mock_cvtColor.return_value = mock_grayscale_image
        grayscale_image = self.processor.convert_to_grayscale()

        # Assert
        mock_imread.assert_called_once_with(self.image_path)
        mock_cvtColor.assert_called_once_with(mock_image, cv2.COLOR_BGR2GRAY)
        self.assertEqual(grayscale_image, mock_grayscale_image)

    @patch("cv2.threshold")
    def test_apply_threshold(self, mock_threshold):
        mock_gray_image = MagicMock()
        mock_threshold.return_value = (127, MagicMock())
        self.processor.apply_threshold(mock_gray_image)

        # Assert
        mock_threshold.assert_called_once_with(
            mock_gray_image, 127, 255, cv2.THRESH_BINARY_INV
        )
        self.assertEqual(
            self.processor.binary_image, mock_threshold.return_value[1]
            )

    @patch("cv2.findContours")
    def test_find_contours(self, mock_findContours):
        mock_binary_image = MagicMock()
        mock_contours = ["contour1", "contour2"]
        mock_findContours.return_value = (mock_contours, None)
        self.processor.binary_image = mock_binary_image
        self.processor.find_contours()

        # Assert
        mock_findContours.assert_called_once_with(
            mock_binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        self.assertEqual(self.processor.contours, mock_contours)


if __name__ == "__main__":
    unittest.main()
