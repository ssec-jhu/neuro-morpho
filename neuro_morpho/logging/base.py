from pathlib import Path

import numpy as np


class Logger:
    """Base logger class to define the interface for experiment logging."""

    def log_scalar(self, name: str, value: float, step: int, train: bool) -> None:
        """Log an informational message."""
        raise NotImplementedError

    def log_triplet(
        self, in_img: np.ndarray, lbl_img: np.ndarray, out_img: np.ndarray, name: str, step: int, train: bool
    ) -> None:
        """Log an image triplet (e.g., for visualization purposes)."""
        raise NotImplementedError

    def log_parameters(self, metrics: dict[str, str | float | int]) -> None:
        """Log a dictionary of hyperparameters."""
        raise NotImplementedError

    def log_code(self, folder: str | Path) -> None:
        """Log the code directory."""
        raise NotImplementedError
