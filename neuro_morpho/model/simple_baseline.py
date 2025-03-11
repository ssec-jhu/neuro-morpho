"""A simple baseline model for testing."""

from pathlib import Path
from typing import override

import gin
import numpy as np
import scipy.ndimage as ndi
import skimage as ski
import skimage.morphology
from tqdm import tqdm

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


@gin.configurable(allowlist=["percentile"])
class SimpleBaseLine(base.BaseModel):
    """A simple baseline model for testing."""

    def __init__(self, percentile: int = 95, name: str|None = None):
        """Initialize the model.

        Args:
            percentile (int, optional): The percentile to use as the threshold. Defaults to 95.
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
    def predict_dir(
        self,
        in_dir: str | Path,
        out_dir: str | Path,
    ) -> None:
        in_dir = Path(in_dir)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        for in_file in tqdm(in_dir.glob("*.pgm")):
            x = ski.io.imread(in_file)[np.newaxis, :, :, np.newaxis]
            y = self.predict(x)[0, :, :, 0]
            ski.io.imsave(
                out_dir / in_file.name, 
                (y*65535).astype(np.int16), 
                check_contrast=False,
            )
            

    @override
    def save(self, path: Path | str) -> None:
        fname = self.name + ".txt"
        with (Path(path) / fname).open("w") as f:
            f.write(str(self.percentile))

    @override
    def load(self, path: Path | str) -> None:
        with Path(path).open("r") as f:
            self.percentile = int(f.read().strip())
