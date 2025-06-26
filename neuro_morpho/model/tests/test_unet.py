"""Tests for the UNet model."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from neuro_morpho.model import unet
from neuro_morpho.model.tiler import Tiler
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


class MockModel:
    """Mock model for testing purposes."""

    def __init__(self, fn: Callable[[np.ndarray], np.ndarray]):
        self.fn = fn

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.fn(x)


class MockOptimizer:
    """Mock optimizer for testing purposes."""

    def step(self):
        pass

    def zero_grad(self):
        pass


class MockLoss:
    def __init__(self, outputs: np.ndarray, targets: np.ndarray):
        self.outputs = outputs
        self.targets = targets
        self.loss_value = np.mean(outputs - targets)

    def item(self) -> float:
        """Return the loss value."""
        return self.loss_value

    def backward(self):
        """Mock backward pass."""
        pass

    def __add__(self, other: "MockLoss") -> "MockLoss":
        """Add two MockLoss instances."""
        return MockLoss(self.outputs + other.outputs, self.targets + other.targets)


class MockLossFn:
    """Mock loss function for testing purposes."""

    def __call__(self, outputs: np.ndarray, targets: np.ndarray) -> float:
        return ("test", MockLoss(outputs, targets))


class MockLogger:
    """Mock logger for testing purposes."""

    def __init__(self):
        self.logs = []

    def log_scalar(self, name: str, value: float, step: int, train: bool = True):
        self.logs.append((name, value, step, train))

    def log_triplet(
        self, in_img: np.ndarray, out_img: np.ndarray, target_img: np.ndarray, name: str, step: int, train: bool
    ):
        self.logs.append((name, in_img, out_img, target_img, step, train))


class MockMetricFn:
    """Mock metric function for testing purposes."""

    def __init__(self, name: str, fn: Callable[[np.ndarray, np.ndarray], float]):
        self.name = name
        self.fn = fn

    def __call__(self, outputs: np.ndarray, targets: np.ndarray) -> float:
        return (self.name, np.mean(outputs - targets))


def test_train_step():
    """Test the train_step method of the UNet model."""
    # Create a mock model and optimizer
    model = MockModel(lambda x: x * 2)
    optimizer = MockOptimizer()
    loss_fn = MockLossFn()

    # Create a dummy input and target
    inputs = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    targets = np.array([[2, 4, 6], [8, 10, 12]], dtype=np.float32)

    # Call the train_step method
    outputs, losses = unet.train_step(model, optimizer, loss_fn, inputs, targets)

    # Check the outputs and losses
    assert np.array_equal(outputs, inputs * 2)
    assert len(losses) == 2
    assert losses[0] == "test"
    assert losses[1].item() == np.mean(outputs - targets)
    assert isinstance(losses[1], MockLoss)


def test_test_step():
    """Test the test_step method of the UNet model."""
    # Create a mock model and optimizer
    model = MockModel(lambda x: x * 2)
    loss_fn = MockLossFn()

    # Create a dummy input and target
    inputs = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    targets = np.array([[2, 4, 6], [8, 10, 12]], dtype=np.float32)

    # Call the train_step method
    outputs, losses = unet.test_step(model, loss_fn, inputs, targets)

    # Check the outputs and losses
    assert np.array_equal(outputs, inputs * 2)
    assert len(losses) == 2
    assert losses[0] == "test"
    assert losses[1].item() == np.mean(outputs - targets)
    assert isinstance(losses[1], MockLoss)


def test_log_metrics():
    """Test the log_metrics method of the UNet model."""
    # Create a mock logger
    logger = MockLogger()
    metric_fs = [
        MockMetricFn("mean", lambda x, y: np.mean(x - y)),
        MockMetricFn("std", lambda x, y: np.std(x - y)),
    ]
    expected_logs = [
        ("mean", 0.0, 1, True),
        ("std", 0.0, 1, True),
    ]

    pred = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    target = np.array([[2, 4, 6], [8, 10, 12]], dtype=np.float32)
    is_train = True
    step = 1

    unet.log_metrics(
        logger=logger,
        metric_fns=metric_fs,
        pred=pred,
        y=target,
        is_train=is_train,
        step=step,
    )

    # Check the logs
    assert len(logger.logs) == len(expected_logs)


def test_log_losses():
    """Test the log_losses method of the UNet model."""
    # Create a mock logger
    logger = MockLogger()
    losses = [
        ("loss1", MockLoss(np.array([1, 2]), np.array([2, 3]))),
        ("loss2", MockLoss(np.array([3, 4]), np.array([4, 5]))),
    ]
    total_loss = MockLoss(np.array([4, 6]), np.array([6, 8]))
    is_train = True
    step = 1

    unet.log_losses(logger=logger, losses=losses, total_loss=total_loss, is_train=is_train, step=step)

    # Check the logs
    assert len(logger.logs) == len(losses) + 1
    assert logger.logs[-1] == ("loss", total_loss.item(), step, is_train)


def test_log_sample():
    """Test the log_sample method of the UNet model."""
    logger = MockLogger()
    in_img = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    out_img = np.array([[2, 4, 6], [8, 10, 12]], dtype=np.float32)
    target_img = np.array([[3, 6, 9], [12, 15, 18]], dtype=np.float32)
    step = 1
    is_train = True
    idx = 0
    unet.log_sample(
        logger=logger,
        x=in_img,
        y=target_img,
        pred=out_img,
        is_train=is_train,
        step=step,
        idx=idx,
    )

    assert len(logger.logs) == 1
    assert logger.logs[0][0] == "triplet"
    assert np.array_equal(logger.logs[0][1], in_img[idx, ...].squeeze())
    assert np.array_equal(logger.logs[0][2], target_img[idx, ...].squeeze())
    assert np.array_equal(logger.logs[0][3], out_img[idx, ...].squeeze())
    assert logger.logs[0][4] == step
    assert logger.logs[0][5] == is_train


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
    tiler = Tiler(tile_size, tile_assembly)
    tiler.get_tiling_attributes(image_size=(256, 256))
    output_tensor = model.predict_proba(input_tensor, tiler)
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
    loaded_model.load_checkpoint(tmp_path)

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


def test_fit_no_epochs(tmp_path: Path):
    """Test the fit method with no epochs."""
    model_id = "test"
    model = unet.UNet(
        n_input_channels=1,
        n_output_channels=1,
        encoder_channels=[64, 128, 256, 512, 1024],
        decoder_channels=[512, 256, 128, 64],
    )
    optimzer = torch.optim.Adam
    model.fit(
        models_dir=tmp_path,
        optimizer=optimzer,
        train_data_loader=object(),  # Mock object for testing
        test_data_loader=object(),  # Mock object for testing
        epochs=0,
        logger=None,
        model_id=model_id,
    )

    assert (tmp_path / model_id / "checkpoints").exists()
    assert (tmp_path / model_id / "model.pt").exists()
