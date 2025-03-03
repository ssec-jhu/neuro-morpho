"""A base class for all models to implement"""

from pathlib import Path

import numpy as np

ERR_NOT_IMPLEMENTED = "The {name} method is not implemented"


class BaseModel:
    """Base class for all models to implement"""

    def fit(self, x: np.ndarray, y: np.ndarray) -> "BaseModel":
        """Fit the model to the data.

        Args:
            x (np.ndarray): The input data should be size of (n_samples, width, height, channels)
            y (np.ndarray): The target data should be size of (n_samples, width, height, channels)

        Returns:
            BaseModel: The fitted model
        """
        raise NotImplementedError(ERR_NOT_IMPLEMENTED.format(name="fit"))

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict the output given the input x

        Args:
            x (np.ndarray): The input data should be size of (n_samples, width, height, channels)

        Returns:
            np.ndarray: The predicted output
        """
        raise NotImplementedError(ERR_NOT_IMPLEMENTED.format(name="predict"))

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Predict a soft version of the output given the input x,

        Args:
            x (np.ndarray): The input data should be size of (n_samples, width, height, channels)

        Returns:
            np.ndarray: The predicted output
        """
        raise NotImplementedError(ERR_NOT_IMPLEMENTED.format(name="predict"))

    def save(self, path: Path | str) -> None:
        """Save the model to the given path.

        Args:
            path (Path|str): The path to save the model
        """
        raise NotImplementedError(ERR_NOT_IMPLEMENTED.format(name="save"))

    def load(self, path: Path | str) -> None:
        """Load the model from the given path.

        Args:
            path (Path|str): The path to load the model
        """
        raise NotImplementedError(ERR_NOT_IMPLEMENTED.format(name="load"))
