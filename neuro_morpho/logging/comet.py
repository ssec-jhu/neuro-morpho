"""Comet.ml logger for experiment tracking."""

import os
from pathlib import Path

import comet_ml
import gin
import matplotlib.pyplot as plt
import numpy as np
from typing_extensions import override

from neuro_morpho.logging import base


@gin.configurable(allowlist=["api_key", "experiment_key", "project_name", "workspace", "disabled"])
class CometLogger(base.Logger):
    """Logger for Comet.ml.

    This logger sends experiment data to Comet.ml for tracking and visualization.
    """

    def __init__(
        self,
        api_key: str | None = None,
        experiment_key: str | None = None,
        project_name: str | None = None,
        workspace: str | None = None,
        auto_param_logging: bool = False,
        auto_metric_logging: bool = False,
        disabled: bool = False,
    ) -> None:
        """Initialize the CometLogger.

        Args:
            api_key (str, optional): The Comet.ml API key. Defaults to None.
            experiment_key (str, optional): The key for an existing experiment. Defaults to None.
            project_name (str, optional): The name of the project. Defaults to None.
            workspace (str, optional): The name of the workspace. Defaults to None.
            auto_param_logging (bool, optional): Whether to automatically log parameters. Defaults to False.
            auto_metric_logging (bool, optional): Whether to automatically log metrics. Defaults to False.
            disabled (bool, optional): Whether to disable the logger. Defaults to False.
        """
        self.experiment = comet_ml.start(
            api_key=api_key or os.getenv("COMET_API_KEY"),
            project_name=project_name,
            workspace=workspace,
            experiment_key=experiment_key,
            experiment_config=comet_ml.ExperimentConfig(
                auto_param_logging=auto_param_logging,
                auto_metric_logging=auto_metric_logging,
                disabled=disabled,
            ),
        )

    @override
    def log_scalar(self, name: str, value: float, step: int, train: bool) -> None:
        ctx = self.experiment.train if train else self.experiment.test
        with ctx():
            self.experiment.log_metric(name, value, step=step)

    @override
    def log_triplet(
        self, in_img: np.ndarray, lbl_img: np.ndarray, out_img: np.ndarray, name: str, step: int, train: bool
    ) -> None:
        fig, (ax_x, ax_pred, ax_y) = plt.subplots(ncols=3, nrows=1, figsize=(30, 10))
        ax_x.imshow(np.log(in_img), cmap="Greys_r")
        ax_x.set_title("log(Input)")
        ax_x.axis("off")
        ax_pred.imshow(out_img, vmin=0, vmax=1, cmap="Greys_r")
        ax_pred.set_title("Predicted")
        ax_pred.axis("off")
        ax_y.imshow(lbl_img, cmap="Greys_r")
        ax_y.set_title("Label")
        ax_y.axis("off")

        ctx = self.experiment.train if train else self.experiment.test
        with ctx():
            self.experiment.log_figure(figure=fig, figure_name=f"{name}", step=step)
        plt.close(fig)

    @override
    def log_parameters(self, metrics: dict[str, str | float | int]) -> None:
        self.experiment.log_parameters(metrics)

    @override
    def log_code(self, folder: Path | str) -> None:
        self.experiment.log_code(folder=folder)
