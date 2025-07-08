# Adapted from:
# https://github.com/namdvt/skeletonization/blob/master/model/unet_att.py
# With the following license:
# MIT License

# Copyright (c) 2025 Nam Nguyen

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Finds the optimal threshold for binarizing a probability map."""

from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import f1_score
from tqdm import tqdm

from neuro_morpho.model.tiler import Tiler


class ThresholdFinder:
    """Finds the optimal threshold for binarizing a probability map.

    This class iterates through a range of thresholds and calculates the F1
    score for each, returning the threshold that maximizes the F1 score.
    """

    def __init__(
        self,
    ):
        super().__init__()

    def find_threshold(
        self,
        pred_dir: str | Path | None = None,
        tar_dir: str | Path | None = None,
        tiler: Tiler = None,
    ) -> float:
        """Find the optimal threshold for binarizing a probability map.

        Args:
            pred_dir (str | Path | None, optional): Directory containing the
                predicted probability maps. Defaults to None.
            tar_dir (str | Path | None, optional): Directory containing
                the ground truth segmentations. Defaults to None.
            tiler (Tiler, optional): Tiler object for tiling the images.
                Defaults to None.

        Returns:
            float: The optimal threshold.
        """
        if pred_dir is None or tar_dir is None:
            raise ValueError("Both prediction and target directories must be provided.")

        preds = list()
        targets = list()

        pred_paths = sorted(list(Path(pred_dir).glob("*_pred.tif")) + list(Path(pred_dir).glob("*_pred.pgm")))
        tar_paths = sorted(list(Path(tar_dir).glob("*.tif")) + list(Path(tar_dir).glob("*.pgm")))
        if not pred_paths or not tar_paths:
            raise ValueError("No images found in one or both of the provided directories.")

        # Ensure the number of images in both directories match
        if len(pred_paths) != len(tar_paths):
            raise ValueError("The number of images in the input and target directories must match.")

        # Read images and targets
        for pred_path, tar_path in zip(pred_paths, tar_paths, strict=False):
            if not pred_path.exists() or not tar_path.exists():
                raise FileNotFoundError(f"Image {pred_path} or target {tar_path} does not exist.")
            # Read the image and target
            pred = cv2.imread(str(pred_path), cv2.IMREAD_UNCHANGED) / 255.0
            target = cv2.imread(str(tar_path), cv2.IMREAD_UNCHANGED)
            target = cv2.convertScaleAbs(target, alpha=255.0 / target.max()) / 255.0
            if pred is None or target is None:
                raise ValueError(
                    f"Could not read image {pred_path} or target {tar_path}. Ensure they are valid image files."
                )
            if tiler is not None:  # tile the both prediction and target images
                pred = tiler.tile_image(pred)
                target = tiler.tile_image(target)
            preds.append(pred)
            targets.append(target)

        preds = np.stack(preds)
        targets = np.stack(targets)

        f1s = list()
        thresholds = np.stack(list(range(40, 80))) / 100
        for threshold in tqdm(thresholds):
            preds_ = preds.copy()
            preds_[preds_ >= threshold] = 1
            preds_[preds_ < threshold] = 0
            f1s.append(f1_score(preds_.reshape(-1), targets.reshape(-1)))
        f1s = np.stack(f1s)
        threshold = thresholds[f1s.argmax()]

        return threshold
