"""Tests for the UNet model."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from neuro_morpho.model import unet
from neuro_morpho.util import get_device


@pytest.mark.parametrize(
    ("fn", "inputs", "expected_output"),
    [
        (lambda x: 2 * x, 2, 4),
        (lambda x: 2 * x, (2, 2), (4, 4)),
    ],
)
def test_apply_tpl(
    fn: Callable,
    inputs: Any | tuple[Any, ...],
    expected_output: Any | tuple[Any, ...],
) -> None:
    """Test the apply_tpl method of the UNet model."""
    assert expected_output == unet.apply_tpl(fn, inputs)


def test_cast_and_move():
    x = torch.tensor([1, 2, 3], dtype=torch.int32)
    x = unet.cast_and_move(x, device="cpu")
    assert x.dtype == torch.float32


def test_detach_and_move():
    inputs = torch.tensor([[1, 2, 3], [3, 4, 5]], dtype=torch.int32)
    outputs = unet.detach_and_move(inputs)
    np.testing.assert_equal(inputs.numpy(), outputs)
    np.testing.assert_equal(inputs.numpy()[0], unet.detach_and_move(inputs, 0))


def test_shapes():
    """Simple end-to-end test for the UNet model."""

    # Create a dummy input tensor with the shape (batch_size, height, width, channels)
    input_tensor = np.random.rand(1, 256, 256, 1).astype(np.float32)

    # Initialize the UNet model
    model = unet.UNet(
        n_input_channels=1,
        n_output_channels=1,
        encoder_channels=[64, 128, 256, 512, 1024],
        decoder_channels=[512, 256, 128, 64],
        device=get_device(),
    )

    output_tensor = model.predict_proba(input_tensor)
    assert output_tensor.shape == (1, 256, 256)


def test_save(tmp_path: Path):
    """Test saving the model."""
    model = unet.UNet(
        n_input_channels=1,
        n_output_channels=1,
        encoder_channels=[64, 128, 256, 512, 1024],
        decoder_channels=[512, 256, 128, 64],
    )

    model_path = tmp_path
    model.save(model_path)
    assert (model_path / "model.pt").exists()


def test_load(tmp_path: Path):
    """Test loading the model."""
    # This test needs to be refined, to maybe check that the state_dict is the same after loading.
    model = unet.UNet(
        n_input_channels=1,
        n_output_channels=1,
        encoder_channels=[64, 128, 256, 512, 1024],
        decoder_channels=[512, 256, 128, 64],
    )

    model.save(tmp_path)
    model_path = tmp_path / "model.pt"

    model_2 = unet.UNet(
        n_input_channels=1,
        n_output_channels=1,
        encoder_channels=[64, 128, 256, 512, 1024],
        decoder_channels=[512, 256, 128, 64],
    )
    model_2.load(model_path)
    assert isinstance(model_2, unet.UNet)
