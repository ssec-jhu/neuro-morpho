"""Tests for the UNet model."""

from pathlib import Path

import numpy as np
import torch
import pytest


from neuro_morpho.model.unet import UNet


def test_shapes():
    """Simple end-to-end test for the UNet model."""

    # Create a dummy input tensor with the shape (batch_size, channels, height, width)
    input_tensor = torch.randn(1, 1, 256, 256)
    input_tensor = np.random.rand(1, 1, 256, 256).astype(np.float32)

    # Initialize the UNet model
    model = UNet(
        n_input_channels=1,
        n_output_channels=1,
        encoder_channels = [64, 128, 256, 512, 1024],
        decoder_channels = [512, 256, 128, 64],
    )

    output_tensor = model.predict_proba(input_tensor)
    assert output_tensor.shape == (1, 256, 256)

def test_predict_dir_raises():
    """Test that predict_dir raises an error when called."""
    model = UNet(
        n_input_channels=1,
        n_output_channels=1,
        encoder_channels = [64, 128, 256, 512, 1024],
        decoder_channels = [512, 256, 128, 64],
    )

    with pytest.raises(NotImplementedError):
        model.predict_dir("dummy_path", "")  # This should raise an error since predict_dir is not implemented.

def test_save(tmp_path:Path):
    """Test saving the model."""
    model = UNet(
        n_input_channels=1,
        n_output_channels=1,
        encoder_channels = [64, 128, 256, 512, 1024],
        decoder_channels = [512, 256, 128, 64],
    )

    model_path = tmp_path
    model.save(model_path)
    assert (model_path / "model.pt").exists()

def test_load(tmp_path:Path):
    """Test loading the model."""
    # This test needs to be refined, to maybe check that the state_dict is the same after loading.
    model = UNet(
        n_input_channels=1,
        n_output_channels=1,
        encoder_channels = [64, 128, 256, 512, 1024],
        decoder_channels = [512, 256, 128, 64],
    )

    model.save(tmp_path)
    model_path = tmp_path / "model.pt"

    model_2 = UNet(
        n_input_channels=1,
        n_output_channels=1,
        encoder_channels = [64, 128, 256, 512, 1024],
        decoder_channels = [512, 256, 128, 64],
    )
    model_2.load(model_path)
    assert isinstance(model_2, UNet)
