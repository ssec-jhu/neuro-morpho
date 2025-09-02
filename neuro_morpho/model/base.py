"""Base class for all models."""

from pathlib import Path

import numpy as np

from neuro_morpho.model.tiler import Tiler

ERR_NOT_IMPLEMENTED = "The {name} method is not implemented"


class BaseModel:
    """Base class for all models.

    This class defines the interface for all models. All models should inherit
    from this class and implement the methods defined here.
    """

    def fit(self, data_dir: str | Path) -> "BaseModel":
        """Fit the model to the data.

        Args:
            data_dir (str|Path): The directory containing the data files to fit the model
                images should have the size (n_samples, channels, height, width)

        Returns:
            BaseModel: The fitted model
        """
        raise NotImplementedError(ERR_NOT_IMPLEMENTED.format(name="fit"))

    def predict_dir(
        self,
        in_dir: str | Path,
        out_dir: str | Path,
        threshold: float,
        mode: str,
        tile_size: tuple[int, int],
        tile_assembly: str,
        binarize: bool,
        fix_breaks: bool,
    ) -> None:
        """Predict the output for all images in the given directory.

        Args:
            in_dir (str|Path): The directory containing the data files to predict
                images should have the size (n_samples, channels, height, width)
            out_dir (str|Path): The directory to save the output
            threshold (float): Use to get the hard prediction (binary output)
            mode (str): The mode of the prediction, can be 'test' or 'infer'
                'test' - runs the model on the test set (same size images) and saves the statistics
                'infer' - runs the model on the inference set (images may be of different size) and saves the output
            tile_size (tuple[int, int]): The size of the tiles to use for tiling the input images
            tile_assembly (str): The method for assembling the tiles, can be 'nn' (nearest neighbor), 'mean', or 'max'
            binarize (bool): Whether to binarize the output
            fix_breaks (bool): Whether to fix breaks in the binarized output
        """
        raise NotImplementedError(ERR_NOT_IMPLEMENTED.format(name="predict_dir"))

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict the output given the input x

        Args:
            x (np.ndarray): The input data should be size of (n_samples, channels, height, width)
            thresh (float): The threshold to use for the prediction

        Returns:
            np.ndarray: The predicted output
        """
        raise NotImplementedError(ERR_NOT_IMPLEMENTED.format(name="predict"))

    def predict_proba(self, x: np.ndarray, tiler: Tiler) -> np.ndarray:
        """Predict a soft version of the output given the input x and tiling params as an option

        Args:
            x (np.ndarray): The input data should be size of (n_samples, channels, height, width)
            tiler (Tiler): The tiler object to use for tiling the input data

        Returns:
            np.ndarray: The predicted output
        """
        raise NotImplementedError(ERR_NOT_IMPLEMENTED.format(name="predict_proba"))

    def find_threshold(
        self,
        in_dir: str | Path,
        out_dir: str | Path,
        model_dir: str | Path,
        model_out_val_y_dir: str | Path) -> float:
        """Predict the output for all images in the given directory.

        Args:
            in_dir (str|Path): The directory containing images (validation set)
            out_dir (str|Path): The directory containing labels (validation set)
            model_dir (str|Path): The directory containing model checkpoints
        """
        raise NotImplementedError(ERR_NOT_IMPLEMENTED.format(name="find_threshold"))

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
