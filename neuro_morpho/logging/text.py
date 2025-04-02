from pathlib import Path
from typing import override

import numpy as np
import plotly.express as px

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
        if not self.triplet_dir.exists():
            self.triplet_dir.mkdir(parents=True, exist_ok=True)

    @override
    def log_scalar(self, name: str, value: float, step: int) -> None:
        with open(self.log_dir / f"{name}.txt", "a") as f:
            f.write(f"{step},{value}\n")

    @override
    def log_triplet(
        self,
        in_img: np.ndarray,
        lbl_img: np.ndarray,
        out_img: np.ndarray,
        name: str,
        step: int,
    ) -> None:
        """Log a triplet of images (input, label, output) to a html file."""
        img_seq = [
            in_img / in_img.max(),
            out_img / out_img.max(),
            lbl_img / lbl_img.max(),
        ]

        fig = px.imshow(
            np.array(img_seq),
            facet_col=0,
            labels={"facet_col": "Image"},
            binary_string=True,
        )
        titles = ["Input Image", "Predicted Image", "Label Image"]
        for i, title in enumerate(titles):
            fig.layout.annotations[i].update(text=title)

        fig.write_html(self.triplet_dir / f"{name}_{step}.html")
