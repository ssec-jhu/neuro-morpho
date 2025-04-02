import numpy as np


class Logger:
    """Base logger class to define the interface for experiment logging."""

    def log_scalar(self, name: str, value: float, step: int) -> None:
        """Log an informational message."""
        raise NotImplementedError

    def log_triplet(self, in_img: np.ndarray, lbl_img: np.ndarray, out_img: np.ndarray, name: str, step: int) -> None:
        """Log an image triplet (e.g., for visualization purposes)."""
        raise NotImplementedError
