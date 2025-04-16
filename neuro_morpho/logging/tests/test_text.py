"""Test cases for the TextLogger class in the neuro_morpho.logging.text module."""

import json
from pathlib import Path

import numpy as np

from neuro_morpho.logging.text import TextLogger


def test_init(tmp_path: Path) -> None:
    """Test that the TextLogger initializes correctly."""
    TextLogger(tmp_path)

    assert (tmp_path / "triplets").exists()
    assert (tmp_path / "train").exists()
    assert (tmp_path / "test").exists()


def test_log_scalar(tmp_path: Path) -> None:
    """Test logging a scalar value."""
    logger = TextLogger(tmp_path)
    logger.log_scalar("test_metric", 0.5, 1, train=True)

    train_file = tmp_path / "train" / "test_metric.txt"
    assert train_file.exists()

    with open(train_file, "r") as f:
        content = f.read().strip()
        assert content == "1,0.5"


def test_log_triplet(tmp_path: Path) -> None:
    """Test logging a scalar value."""
    logger = TextLogger(tmp_path)

    in_img = (np.eye(10, 10) + 1) * 5
    lbl_img = np.eye(10, 10) + 1
    out_img = (np.eye(10, 10) + 1) * 0.5
    train_triplet_dir = tmp_path / "triplets" / "train"
    logger.log_triplet(in_img, lbl_img, out_img, "test_triplet", 1, train=True)
    assert train_triplet_dir.exists()
    assert (train_triplet_dir / "test_triplet_1.png").exists()


def test_log_parameters(tmp_path: Path) -> None:
    """Test logging parameters."""
    logger = TextLogger(tmp_path)
    metrics = {"learning_rate": 0.001, "batch_size": 32}
    logger.log_parameters(metrics)

    param_file = tmp_path / "parameters.json"
    assert param_file.exists()

    with open(param_file, "r") as f:
        restored_params = json.load(f)
    assert restored_params == metrics


def test_log_code(tmp_path: Path) -> None:
    """Test logging code directory."""
    logger = TextLogger(tmp_path)
    code_folder = tmp_path / "tmp_code"
    code_folder.mkdir(parents=True, exist_ok=True)

    # Create a dummy file in the code folder
    (code_folder / "dummy.py").write_text("print('Hello, World!')")

    logger.log_code(code_folder)

    logged_code_dir = tmp_path / "code"
    assert logged_code_dir.exists()
    assert (logged_code_dir / "dummy.py").exists()
