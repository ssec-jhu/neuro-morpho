"""Test the simple baseline model."""

import os
from pathlib import Path

import numpy as np

from neuro_morpho.model import simple_baseline


def test_make_binary() -> None:
    """Test the make_binary function."""
    # Test with a simple array and a percentile
    arr = np.array([[0.1, 0.5], [0.3, 0.7]]).reshape(1, 2, 2)
    percentile = 50
    binary_arr = simple_baseline.make_binary(arr, percentile)

    # Check that values above the 50th percentile are set to 1
    assert np.all(binary_arr[arr >= np.percentile(arr, percentile)] == 1)
    # Check that values below the 50th percentile are set to 0
    assert np.all(binary_arr[arr < np.percentile(arr, percentile)] == 0)


def test_simple_baseline_init() -> None:
    """Test the SimpleBaseLine model initialization."""
    model = simple_baseline.SimpleBaseLine(percentile=90, name="test_model")
    assert model.percentile == 90
    assert model.name == "test_model"


def test_simple_baseline_fit() -> None:
    """Test the fit method of SimpleBaseLine model."""
    model = simple_baseline.SimpleBaseLine(percentile=90)
    trained_model = model.fit("training_x_dir", "training_y_dir", "testing_x_dir", "testing_y_dir")

    assert isinstance(trained_model, simple_baseline.SimpleBaseLine)
    assert trained_model.percentile == 90
    assert trained_model.name == "simple_base_line"


def test_simple_baseline_predict() -> None:
    """Test the predict method of SimpleBaseLine model."""
    model = simple_baseline.SimpleBaseLine(percentile=90)
    x = np.array([[0.1, 0.5], [0.3, 0.7]]).reshape(1, 2, 2, 1)
    predictions = model.predict(x)

    # Check that the predictions are binary
    assert np.all(np.isin(predictions, [0, 1]))
    # Check that the shape of predictions matches input
    assert predictions.shape == x.shape


def test_save(tmp_path: Path) -> None:
    """Test the save method of SimpleBaseLine model."""
    model = simple_baseline.SimpleBaseLine(percentile=90)
    model.save(tmp_path)

    # Check if the file exists
    assert os.path.exists(tmp_path / (model.name + ".txt"))


def test_load(tmp_path: Path) -> None:
    """Test the load method of SimpleBaseLine model."""
    model = simple_baseline.SimpleBaseLine(percentile=90)
    model.save(tmp_path)

    new_model = simple_baseline.SimpleBaseLine()
    new_model.load(tmp_path/(model.name + ".txt"))

    # Check that the loaded model has the same percentile
    assert new_model.percentile == 90
    assert new_model.name == "simple_base_line"
