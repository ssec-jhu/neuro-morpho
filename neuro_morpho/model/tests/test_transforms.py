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


def test_DownSample() -> None:
    """Test the downsample function."""
    arr = torch.from_numpy(
        np.array([[1, 2, 1, 2], [1, 2, 1, 2], [3, 4, 3, 4], [3, 4, 3, 4]], dtype=np.float32).reshape(1, 4, 4)
    )
    downsampled_arr = transforms.DownSample(in_size=(4, 4), factors=0.5)(arr)

    # Check that the shape is correct after downsampling
    assert downsampled_arr.shape == (1, 2, 2)
    # Test with multiple factors
    downsampled_arr_multi = transforms.DownSample(in_size=(4, 4), factors=[0.5, 0.25])(arr)
    assert downsampled_arr_multi[0].shape == (1, 2, 2)
    assert downsampled_arr_multi[1].shape == (1, 1, 1)
