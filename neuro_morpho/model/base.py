"""A base class for all models to implement"""

from pathlib import Path

import numpy as np

from neuro_morpho.model.tiler import Tiler

ERR_NOT_IMPLEMENTED = "The {name} method is not implemented"


class BaseModel:
    """Base class for all models to implement"""

    def fit(self, data_dir: str | Path) -> "BaseModel":
        """Fit the model to the data.

        Args:
            data_dir (str|Path): The directory containing the data files to fit the model
                images should have the size (n_samples, width, height, channels)

        Returns:
            BaseModel: The fitted model
        """
        raise NotImplementedError(ERR_NOT_IMPLEMENTED.format(name="fit"))

    def predict_dir(self, in_dir: str | Path, out_dir: str | Path, tiler: Tiler, binarize: bool) -> None:
        """Predict the output for all images in the given directory.

        Args:
            in_dir (str|Path): The directory containing the data files to predict
                images should have the size (n_samples, channels, width, height)
            out_dir (str|Path): The directory to save the output
        """
        raise NotImplementedError(ERR_NOT_IMPLEMENTED.format(name="predict_dir"))

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict the output given the input x

        Args:
            x (np.ndarray): The input data should be size of (n_samples, channels, width, height)
            thresh (float): The threshold to use for the prediction

        Returns:
            np.ndarray: The predicted output
        """
        raise NotImplementedError(ERR_NOT_IMPLEMENTED.format(name="predict"))

    def predict_proba(self, x: np.ndarray, tiler: Tiler) -> np.ndarray:
        """Predict a soft version of the output given the input x and tiling params as an option

        Args:
            x (np.ndarray): The input data should be size of (n_samples, channels, width, height)
            tiler (Tiler): The tiler object to use for tiling the input data

        Returns:
            np.ndarray: The predicted output
        """
        raise NotImplementedError(ERR_NOT_IMPLEMENTED.format(name="predict_proba"))

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
