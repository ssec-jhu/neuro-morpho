from pathlib import Path
from typing import override

import matplotlib.pyplot as plt
import numpy as np

from neuro_morpho.logging import base
from neuro_morpho.logging.plots import plot_triplet


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
            (self.triplet_dir / subdir).mkdir(parents=True, exist_ok=True)

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
        fig = plot_triplet(
            in_img=in_img,
            lbl_img=lbl_img,
            out_img=out_img,
        )
        fig.savefig(
            self.triplet_dir / subdir / f"{name}_{step}.png",
            bbox_inches="tight",
            pad_inches=0.1,
        )
        plt.close(fig)
