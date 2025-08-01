"""Test the neuro_morpho model loss functions."""

import pytest
import torch

from neuro_morpho.model import loss

ONES = torch.ones((1, 1, 2, 2), dtype=torch.float32)
ZEROS = torch.zeros((1, 1, 2, 2), dtype=torch.float32)
HALF = torch.ones((1, 1, 2, 2), dtype=torch.float32) * 0.5


@pytest.mark.parametrize(
    ("inputs", "targets"),
    [
        (ONES, ONES),
        (ZEROS, ZEROS),
        (HALF, HALF),
        (ONES, ZEROS),
        (ZEROS, ONES),
        (HALF, ONES),
        (HALF, ZEROS),
    ],
)
def test_weighted_focal_loss(inputs: torch.Tensor, targets: torch.Tensor) -> None:
    """Test the WeightedFocalLoss class."""

    loss_fn = loss.WeightedFocalLoss(alpha=0.25, gamma=2)
    inputs = torch.ones((1, 1, 2, 2), dtype=torch.float32) * 0.5

    name_loss, loss_value = loss_fn(inputs, targets)
    assert name_loss == "weighted_focal_loss"
    assert isinstance(loss_value, torch.Tensor)
    assert len(loss_value.shape) == 0
    assert not torch.isnan(loss_value).any()
    assert loss_value.item() >= 0.0


@pytest.mark.parametrize(
    ("inputs", "targets"),
    [
        (ONES, ONES),
        (ZEROS, ZEROS),
        (HALF, HALF),
        (ONES, ZEROS),
        (ZEROS, ONES),
        (HALF, ONES),
        (HALF, ZEROS),
    ],
)
def test_dice_loss(inputs: torch.Tensor, targets: torch.Tensor) -> None:
    """Test the SigmoidDiceLoss class."""

    loss_fn = loss.SigmoidDiceLoss(smooth=1.0)
    preds = torch.ones((1, 1, 2, 2), dtype=torch.float32) * 0.5

    name_loss, loss_value = loss_fn(preds, targets)
    assert name_loss == "dice_loss"
    assert isinstance(loss_value, torch.Tensor)
    assert len(loss_value.shape) == 0
    assert not torch.isnan(loss_value).any()
    assert loss_value.item() >= 0.0


def test_weighted_map() -> None:
    """Test the WeightedMap class."""

    loss_fn = loss.WeightedFocalLoss(alpha=0.25, gamma=2)
    coefs = [0.5, 0.5]
    weighted_map_loss = loss.WeightedMap(loss_fn, coefs)

    preds = [torch.ones((1, 1, 2, 2), dtype=torch.float32) * 0.5, torch.ones((1, 1, 2, 2), dtype=torch.float32) * 0.5]
    targets = [torch.ones((1, 1, 2, 2), dtype=torch.float32), torch.ones((1, 1, 2, 2), dtype=torch.float32)]

    name_loss, loss_value = weighted_map_loss(preds, targets)
    assert name_loss == "weighted_focal_loss"
    assert isinstance(loss_value, torch.Tensor)
    assert len(loss_value.shape) == 0
    assert not torch.isnan(loss_value).any()
    assert loss_value.item() >= 0.0


def test_combined_loss() -> None:
    """Test the CombinedLoss class."""
    loss_fn1 = loss.WeightedFocalLoss(alpha=0.25, gamma=2)
    loss_fn2 = loss.SigmoidDiceLoss(smooth=1.0)
    weights = [0.5, 0.5]
    combined_loss = loss.CombinedLoss(weights, [loss_fn1, loss_fn2])

    preds = torch.ones((1, 1, 2, 2), dtype=torch.float32) * 0.5
    targets = torch.ones((1, 1, 2, 2), dtype=torch.float32)

    names, losses = zip(*combined_loss(preds, targets))
    assert len(names) == 2
    assert len(losses) == 2
    assert names[0] == "weighted_focal_loss"
    assert names[1] == "dice_loss"
    assert not torch.isnan(losses[0]).any()
    assert not torch.isnan(losses[1]).any()
    assert losses[0].item() >= 0.0
    assert losses[1].item() >= 0.0
