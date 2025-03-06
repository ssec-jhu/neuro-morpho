"""A simple baseline model for testing."""

from pathlib import Path
from typing import override

import numpy as np
import scipy.ndimage as ndi
import skimage.morphology

import neuro_morpho.model.base as base


def make_binary(
    x: np.ndarray,
    percentile: int,
) -> np.ndarray:
    """Make the input images binary based on a threshold.

    Args:
        x (np.ndarray): The input data should be size of
            (n_samples, width, height), channels should be one.
        percentile (int): The percentile to use as the threshold
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


class SimpleBaseLine(base.BaseModel):
    """A simple baseline model for testing."""

    def __init__(self, percentile: int = 95):
        """Initialize the model.

        Args:
            percentile (int, optional): The percentile to use as the threshold. Defaults to 95.
        """
        self.percentile = percentile

    @override
    def fit(
        self,
        data_dir: Path | str,
    ) -> "SimpleBaseLine":
        return self

    @override
    def predict(
        self,
        x: np.ndarray,
    ) -> np.ndarray:
        x = np.squeeze(x, axis=-1)
        x = make_binary(x, self.percentile)
        return np.expand_dims(x, axis=-1)

    @override
    def save(self, path: Path | str) -> None:
        with Path(path).open("w") as f:
            f.write(str(self.percentile))

    @override
    def load(self, path: Path | str) -> None:
        with Path(path).open("r") as f:
            self.percentile = int(f.read().strip())
