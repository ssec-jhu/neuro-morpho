import os
from typing import override

import comet_ml
import gin
import numpy as np
import plotly.express as px

from neuro_morpho.logging import base


@gin.configurable(allowlist=["api_key", "project_name", "workspace"])
class CometLogger(base.Logger):
    """Logger class for logging metrics and images to Comet.ml."""

    def __init__(
        self,
        api_key: str | None = None,
        project_name: str | None = None,
        workspace: str | None = None,
        auto_param_logging: bool = False,
        auto_metric_logging: bool = False,
        disabled: bool = False,
    ) -> None:
        """Initialize the CometLogger with a comet.ml experiment.

        Args:
            experiment (comet_ml.Experiment): The comet.ml experiment object.
        """
        self.experiment = comet_ml.Experiment(
            api_key=api_key or os.getenv("COMET_API_KEY"),
            project_name=project_name,
            workspace=workspace,
            auto_param_logging=auto_param_logging,
            auto_metric_logging=auto_metric_logging,
            disabled=disabled,
        )

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
