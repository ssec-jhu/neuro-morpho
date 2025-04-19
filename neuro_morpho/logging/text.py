import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from typing_extensions import override

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

    @override
    def log_parameters(self, metrics: dict[str, str | float | int]) -> None:
        with open(self.log_dir / "parameters.json", "a") as f:
            json.dump(metrics, f)

    @override
    def log_code(self, folder: str | Path) -> None:
        code_dir = self.log_dir / "code"
        code_dir.mkdir(parents=True, exist_ok=True)
        for file in Path(folder).rglob("*"):
            if file.is_file():
                relative_path = file.relative_to(folder)
                target_path = code_dir / relative_path
                target_path.parent.mkdir(parents=True, exist_ok=True)
                target_path.write_text(file.read_text())
