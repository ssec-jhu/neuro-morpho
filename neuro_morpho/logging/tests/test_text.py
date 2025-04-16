"""Test cases for the TextLogger class in the neuro_morpho.logging.text module."""

from pathlib import Path

import numpy as np

from neuro_morpho.logging.text import TextLogger


def test_text_logger_init(tmp_path: Path) -> None:
    """Test that the TextLogger initializes correctly."""
    TextLogger(tmp_path)

    assert (tmp_path / "triplets").exists()
    assert (tmp_path / "train").exists()
    assert (tmp_path / "test").exists()


def test_text_logger_log_scalar(tmp_path: Path) -> None:
    """Test logging a scalar value."""
    logger = TextLogger(tmp_path)
    logger.log_scalar("test_metric", 0.5, 1, train=True)

    train_file = tmp_path / "train" / "test_metric.txt"
    assert train_file.exists()

    with open(train_file, "r") as f:
        content = f.read().strip()
        assert content == "1,0.5"


def test_text_logger_log_triplet(tmp_path: Path) -> None:
    """Test logging a scalar value."""
    logger = TextLogger(tmp_path)

    in_img = (np.eye(10, 10) + 1) * 5
    lbl_img = np.eye(10, 10) + 1
    out_img = (np.eye(10, 10) + 1) * 0.5
    train_triplet_dir = tmp_path / "triplets" / "train"
    logger.log_triplet(in_img, lbl_img, out_img, "test_triplet", 1, train=True)
    assert train_triplet_dir.exists()
    assert (train_triplet_dir / "test_triplet_1.png").exists()
