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
    input_tensor = np.random.rand(1, 1, 256, 256).astype(np.float32)

    # Initialize the UNet model
    model = unet.UNet(
        n_input_channels=1,
        n_output_channels=1,
        encoder_channels=[64, 128, 256, 512, 1024],
        decoder_channels=[512, 256, 128, 64],
        device=get_device(),
    )
    tile_size = 128
    tile_assembly = "mean"
    output_tensor = model.predict_proba(input_tensor, tile_size, tile_assembly)
    assert output_tensor.shape == (1, 1, 256, 256)


def test_save(tmp_path: Path):
    """Test saving the model."""
    model = unet.UNet(
        n_input_channels=1,
        n_output_channels=1,
        encoder_channels=[64, 128, 256, 512, 1024],
        decoder_channels=[512, 256, 128, 64],
    )

    model_path = tmp_path / "model.pt"
    model.save(model_path)
    assert model_path.exists()


def test_load(tmp_path: Path):
    """Test loading the model."""
    # This test needs to be refined, to maybe check that the state_dict is the same after loading.
    model = unet.UNet(
        n_input_channels=1,
        n_output_channels=1,
        encoder_channels=[64, 128, 256, 512, 1024],
        decoder_channels=[512, 256, 128, 64],
    )

    model_path = tmp_path / "model.pt"
    model.save(model_path)

    model_2 = unet.UNet(
        n_input_channels=1,
        n_output_channels=1,
        encoder_channels=[64, 128, 256, 512, 1024],
        decoder_channels=[512, 256, 128, 64],
    )
    model_2.load(model_path)
    assert isinstance(model_2, unet.UNet)


def test_save_checkpoint_single(tmp_path: Path):
    """Test saving a checkpoint."""
    model = unet.UNet(
        n_input_channels=1,
        n_output_channels=1,
        encoder_channels=[64, 128, 256, 512, 1024],
        decoder_channels=[512, 256, 128, 64],
    )
    step = 0
    n_checkpoints = 5
    expected_path = tmp_path / f"checkpoint_{step}.pt"

    model.save_checkpoint(tmp_path, n_checkpoints, step)

    assert expected_path.exists()
    assert expected_path.is_file()


def test_save_checkpoint_multiple(tmp_path: Path):
    """Test saving a checkpoint."""
    model = unet.UNet(
        n_input_channels=1,
        n_output_channels=1,
        encoder_channels=[64, 128, 256, 512, 1024],
        decoder_channels=[512, 256, 128, 64],
    )

    n_checkpoints = 3

    for i in range(n_checkpoints + 1):
        expected_path = tmp_path / f"checkpoint_{i}.pt"
        model.save_checkpoint(tmp_path, n_checkpoints, i)
        assert expected_path.exists()
        assert expected_path.is_file()

    # the oldest checkpoint should have been removed
    assert not (tmp_path / "checkpoint_0.pt").exists()


def test_load_checkpoint(tmp_path: Path):
    """Test loading a checkpoint."""
    model = unet.UNet(
        n_input_channels=1,
        n_output_channels=1,
        encoder_channels=[64, 128, 256, 512, 1024],
        decoder_channels=[512, 256, 128, 64],
    )

    step = 0
    model_path = tmp_path / f"checkpoint_{step}.pt"
    model.save(model_path)

    loaded_model = unet.UNet(
        n_input_channels=1,
        n_output_channels=1,
        encoder_channels=[64, 128, 256, 512, 1024],
        decoder_channels=[512, 256, 128, 64],
    )
    loaded_model.load_checkpoint(model_path)

    for k in model.model.state_dict().keys():
        assert k in loaded_model.model.state_dict()
        assert model.model.state_dict()[k].shape == loaded_model.model.state_dict()[k].shape


def test_load_checkpoint_invalid_path():
    """Test loading a checkpoint with an invalid path."""
    model = unet.UNet(
        n_input_channels=1,
        n_output_channels=1,
        encoder_channels=[64, 128, 256, 512, 1024],
        decoder_channels=[512, 256, 128, 64],
    )

    with pytest.raises(FileNotFoundError):
        model.load_checkpoint("invalid_path")
