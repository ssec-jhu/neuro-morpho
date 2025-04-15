from pathlib import Path
from typing import override

import matplotlib.pyplot as plt
import numpy as np

from neuro_morpho.logging import base


class TextLogger(base.Logger):
    """Logger class for logging metrics and images to text files."""

    def __init__(self, log_dir: Path | str) -> None:
        """Initialize the TextLogger with a log file.

        Args:
            log_file (str): The path to the log file.
        """
        self.log_dir = Path(log_dir)
        self.triplet_dir = self.log_dir / "triplets"
        for subdir in ["train", "test"]:
            (self.log_dir / subdir).mkdir(parents=True, exist_ok=True)
        if not self.triplet_dir.exists():
            self.triplet_dir.mkdir(parents=True, exist_ok=True)

    @override
    def log_scalar(self, name: str, value: float, step: int, train: bool) -> None:
        subdir = "train" if train else "test"
        with open(self.log_dir / subdir / f"{name}.txt", "a") as f:
            f.write(f"{step},{value}\n")

    @override
    def log_triplet(
        self,
        in_img: np.ndarray,
        lbl_img: np.ndarray,
        out_img: np.ndarray,
        name: str,
        step: int,
        train: bool,
    ) -> None:
        subdir = "train" if train else "test"

        _, (ax_x, ax_pred, ax_y) = plt.subplots(ncols=3, nrows=1, figsize=(30, 10))
        ax_x.imshow(np.log(in_img), cmap="Greys_r")
        ax_x.set_title("log(Input)")
        ax_x.axis("off")
        ax_pred.imshow(out_img, vmin=0, vmax=1, cmap="Greys_r")
        ax_pred.set_title("Predicted")
        ax_pred.axis("off")
        ax_y.imshow(lbl_img, cmap="Greys_r")
        ax_y.set_title("Label")
        ax_y.axis("off")
        plt.savefig(
            self.triplet_dir / subdir / f"{name}_{step}.png",
            bbox_inches="tight",
            pad_inches=0.1,
        )
