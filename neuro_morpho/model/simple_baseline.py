"""A simple baseline model for testing."""

from pathlib import Path

import gin
import numpy as np
import scipy.ndimage as ndi
import skimage as ski
import skimage.morphology
from tqdm import tqdm
from typing_extensions import override

import neuro_morpho.model.base as base


def make_binary(
    x: np.ndarray,
    percentile: int,
) -> np.ndarray:
    """Binarize an image based on a percentile threshold.

    This function thresholds the input image `x` at the given `percentile`
    and then skeletonizes the result.

    Args:
        x (np.ndarray): The input image. Should be of shape (n_samples, width, height).
        percentile (int): The percentile to use as the threshold.

    Returns:
        np.ndarray: The binarized and skeletonized image.
    """
    thresholds = np.percentile(x, percentile, axis=(1, 2), keepdims=True)  # (n, 1, 1)
    binarized = np.greater_equal(x, thresholds)  # (n, w, h)

    for i in range(binarized.shape[0]):
        lbls = ndi.label(binarized[i])[0]
        ids, counts = np.unique(binarized[i].flatten(), return_counts=True)
        # exclude the background which will the largest component
        biggest_component = ids[1:][np.argmax(counts[1:])]
        binarized[i] = skimage.morphology.skeletonize(lbls == biggest_component)

    return binarized


@gin.configurable(allowlist=["percentile"])
class SimpleBaseLine(base.BaseModel):
    """A simple baseline model for image segmentation.

    This model binarizes the input image based on a percentile threshold and
    then skeletonizes the result.
    """

    def __init__(self, percentile: int = 95, name: str | None = None):
        """Initialize the model.

        Args:
            percentile (int, optional): The percentile to use as the threshold. Defaults to 95.
            name (str, optional): The name of the model. Defaults to None.
        """
        self.percentile = percentile
        self.name = name or "simple_base_line"

    @override
    def fit(
        self,
        training_x_dir: Path | str,
        training_y_dir: Path | str,
        testing_x_dir: Path | str,
        testing_y_dir: Path | str,
    ) -> "SimpleBaseLine":
        """This model does not require fitting, so this method just returns self."""
        return self

    @override
    def predict(
        self,
        x: np.ndarray,
    ) -> np.ndarray:
        """Predict the segmentation for the input image."""
        x = np.squeeze(x, axis=-1)
        x = make_binary(x, self.percentile)
        return np.expand_dims(x, axis=-1)

    @override
    def predict_dir(
        self,
        in_dir: str | Path,
        out_dir: str | Path,
        tile_size: int = 512,
        tile_assembly: str = "mean",
    ) -> None:
        """Predict the segmentation for all images in a directory."""
        in_dir = Path(in_dir)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        in_files = [f for ext in ("*.pgm", "*.tif") for f in in_dir.glob(ext)]
        for in_file in tqdm(in_files, desc="Predicting"):
            x = ski.io.imread(in_file)[np.newaxis, :, :, np.newaxis]
            y = self.predict(x)[0, :, :, 0]
            ski.io.imsave(
                out_dir / in_file.name,
                (y * 65535).astype(np.int16),
                check_contrast=False,
            )

    @override
    def save(self, path: Path | str) -> None:
        """Save the model's percentile threshold to a file."""
        fname = self.name + ".txt"
        with (Path(path) / fname).open("w") as f:
            f.write(str(self.percentile))

    @override
    def load(self, path: Path | str) -> None:
        """Load the model's percentile threshold from a file."""
        with Path(path).open("r") as f:
            self.percentile = int(f.read().strip())
