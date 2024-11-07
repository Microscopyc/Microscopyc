from unittest.mock import patch, MagicMock
import torch
from Image_identifier.image_identifier import BacteriaIdentifier


# 1. Test Initialization of BacteriaIdentifier class
def test_init():
    identifier = BacteriaIdentifier(num_strains=5, im_shape=(224, 224))
    assert identifier.num_strains == 5
    assert isinstance(identifier.model, torch.nn.Module)
    # Check if the transform was initialized:
    assert identifier.transform is not None


# 2. Test Load Model
@patch("torchvision.models.shufflenet_v2_x0_5")
def test_load_model(mock_shufflenet):
    # Mock to avoid actual loading
    mock_model = MagicMock()
    mock_shufflenet.return_value = mock_model

    # Ensure load_model is called only once
    identifier = BacteriaIdentifier(num_strains=5)
    # Ensure load_model was called only once
    mock_shufflenet.assert_called_once_with(weights="IMAGENET1K_V1")
    # Ensure the correct model was assigned
    assert identifier.model == mock_model
    # Test the load_model method separately
    model = identifier.load_model()
    # Check if the number of output features is correct
    assert (
        model.fc.out_features == 5
    )


# 5. Test if Model Loading Works on Initialization
@patch("torchvision.models.shufflenet_v2_x0_5")
def test_model_loading_on_initialization(mock_shufflenet):
    # Mock the model loading to avoid actual loading
    mock_model = MagicMock()
    mock_shufflenet.return_value = mock_model
    identifier = BacteriaIdentifier(num_strains=5)

    # Check that the model was loaded during initialization
    mock_shufflenet.assert_called_once_with(weights="IMAGENET1K_V1")
    # Check if the correct model was assigned
    assert identifier.model == mock_model
