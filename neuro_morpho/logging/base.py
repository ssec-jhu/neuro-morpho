"""Base logger class for experiment logging."""

from pathlib import Path

import numpy as np


class Logger:
    """Base logger class to define the interface for experiment logging."""

    def log_scalar(self, name: str, value: float, step: int, train: bool) -> None:
        """Log a scalar value.

        Args:
            name (str): The name of the scalar.
            value (float): The value of the scalar.
            step (int): The current step.
            train (bool): Whether this is a training or testing metric.
        """
        raise NotImplementedError

    def log_triplet(
        self, in_img: np.ndarray, lbl_img: np.ndarray, out_img: np.ndarray, name: str, step: int, train: bool
    ) -> None:
        """Log an image triplet (input, label, output).

        Args:
            in_img (np.ndarray): The input image.
            lbl_img (np.ndarray): The label image.
            out_img (np.ndarray): The output image.
            name (str): The name of the triplet.
            step (int): The current step.
            train (bool): Whether this is a training or testing metric.
        """
        raise NotImplementedError

    def log_parameters(self, metrics: dict[str, str | float | int]) -> None:
        """Log a dictionary of hyperparameters.

        Args:
            metrics (dict): The dictionary of hyperparameters.
        """
        raise NotImplementedError

    def log_code(self, folder: str | Path) -> None:
        """Log the code in the given folder.

        Args:
            folder (str | Path): The folder containing the code to log.
        """
        raise NotImplementedError
