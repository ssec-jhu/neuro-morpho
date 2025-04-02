from typing import override

import comet_ml
import numpy as np
import plotly.express as px

from neuro_morpho.logging import base


class CometLogger(base.Logger):
    """Logger class for logging metrics and images to Comet.ml."""

    def __init__(self, experiment: comet_ml.Experiment) -> None:
        """Initialize the CometLogger with a Comet.ml experiment.

        Args:
            experiment (comet_ml.Experiment): The Comet.ml experiment object.
        """
        self.experiment = experiment

    @override
    def log_scalar(self, name: str, value: float, step: int) -> None:
        self.experiment.log_metric(name, value, step=step)

    @override
    def log_triplet(self, in_img: np.ndarray, lbl_img: np.ndarray, out_img: np.ndarray, name: str, step: int) -> None:
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

        self.experiment.log_figure(
            figure=fig,
            figure_name=f"{name}.html",
            step=step,
        )
