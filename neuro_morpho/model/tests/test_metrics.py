"""Test the metrics module."""

import numpy as np

from neuro_morpho.model import metrics


def test_accuracy():
    """Test the accuracy metric."""
    pred = np.array([0.8, 0.6, 0.4, 0.9, 0.2], dtype=np.float32)
    lbl = np.array([1, 1, 1, 1, 1], dtype=np.float32)
    threshold = 0.5

    name, value = metrics.accuracy(pred, lbl, threshold)
    assert name == "accuracy"
    assert value == 0.6  # 3 out of 5 predictions are correct


def test_class_accuracy():
    """Test the class accuracy metric."""
    pred = np.array([0.8, 0.6, 0.4, 0.9, 0.2], dtype=np.float32)
    lbl = np.array([1, 1, 1, 1, 0], dtype=np.float32)
    class_idx = 1
    threshold = 0.5

    name, value = metrics.class_accuracy(pred, lbl, class_idx, threshold)
    assert name == "class_1_accuracy"
    assert value == 0.75  # 3 out of 4 class 1 predictions are correct
