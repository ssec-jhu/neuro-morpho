"""Tests for transforms module."""

import numpy as np
import torch

from neuro_morpho.model import transforms


def test_Standardize() -> None:
    """Test the standardize function."""
    arr = torch.from_numpy(np.array([[1, 2], [3, 4]], dtype=np.float32).reshape(1, 1, 2, 2))
    standardized_arr = transforms.Standardize()(arr)

    # Check that the mean is close to 0 and std is close to 1
    assert abs(standardized_arr.mean()) < 1e-7
    assert abs(standardized_arr.std() - 1) < 1


def test_Norm2One() -> None:
    """Test the normalize function."""
    arr = torch.from_numpy(np.array([[1, 2], [3, 4]], dtype=np.float32).reshape(1, 1, 2, 2))
    normalized_arr = transforms.Norm2One()(arr)

    # Check that the min is 0 and max is 1
    assert abs(normalized_arr.min()) == 1 / 4
    assert abs(normalized_arr.max() - 1) < 1e-7
