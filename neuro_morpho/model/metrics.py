import warnings
from typing import Callable

import gin
import numpy as np

PRED = np.ndarray
LBL = np.ndarray
METRIC_FN = Callable[[PRED, LBL], tuple[str, float]]


@gin.configurable(allowlist=["threshold"])
def accuracy(pred: PRED, lbl: LBL, threshold: float) -> tuple[str, float]:
    """Calculate the accuracy of predictions.

    Args:
        x (np.ndarray): The true labels.

    Returns:
        float: The accuracy as a percentage.
    """
    pred_binary = pred >= threshold
    correct_predictions = np.sum(pred_binary == lbl)
    total_predictions = len(lbl)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        accuracy_value = correct_predictions / total_predictions
    return "accuracy", accuracy_value  # Return as percentage


@gin.configurable(allowlist=["class_idx", "threshold"])
def class_accuracy(pred: PRED, lbl: LBL, class_idx: int, threshold: float) -> tuple[str, float]:
    """Calculate the class-wise accuracy of predictions.

    Args:
        pred (np.ndarray): The predicted labels.
        lbl (np.ndarray): The true labels.
        threshold (float): The threshold for binary classification.

    Returns:
        float: The class-wise accuracy as a percentage.
    """
    mask = lbl == class_idx
    _, val = accuracy(pred[mask], lbl[mask], threshold)

    return f"class_{class_idx}_accuracy", val
