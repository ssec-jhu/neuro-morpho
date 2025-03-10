"""Reports used for summarizing the results of the analysis."""

import re
from pathlib import Path
from typing import Callable

import gin
import matplotlib.pyplot as plt
import pandas as pd

import neuro_morpho.reports.plots as nm_plots

ERR_NOT_IMPLEMENTED = "The {name} method is not implemented"
ERR_COLS_DONT_MATCH = "The columns in the model and label dataframes do not match."
ERR_MISSING_FNAME_COL = "The label data does not have a filename column."

MODEL_OUT_DIR = str | Path
LABEL_DIR = str | Path
REPORT_DIR = str | Path
report_fn = Callable[[MODEL_OUT_DIR, LABEL_DIR, REPORT_DIR], None]


NOISE_LEVEL_PAT = re.compile(r"sbr-\d|SBR-\d")


def transform_filename_to_group(filename: str) -> str:
    group = "Label"
    if noise_level := NOISE_LEVEL_PAT.search(filename):
        group = f"Model({noise_level.group()[-1]})"

    return group


@gin.register
def noboxplot_summary(
    model_out_dir: MODEL_OUT_DIR,
    label_dir: LABEL_DIR,
    report_dir: REPORT_DIR,
) -> None:
    """Generate a summary of the results using a noboxplot-esque plot."""
    # These are pandas dataframes the column `filename` and additional
    # columns for resulting statistics that we want to compute notboxplots for.
    aggregated_model = pd.read_csv(model_out_dir / "aggregated_results.csv")
    aggregated_label = pd.read_csv(label_dir / "aggregated_results.csv")

    if aggregated_label.columns != aggregated_model.columns:
        raise ValueError(ERR_COLS_DONT_MATCH)

    if "filename" not in aggregated_model.columns:
        raise ValueError(ERR_MISSING_FNAME_COL)

    # we can overwrite the filename column in the labels because it doesn't
    # have different levels of noise
    aggregated_label["filename"] = "Label"

    plotting_df = pd.concat([aggregated_model, aggregated_label])
    plotting_df["filename"] = plotting_df["filename"].apply(transform_filename_to_group)

    plotting_columns = list(plotting_df.columns)
    plotting_columns.remove("filename")

    # Generate the summary
    f, axes = plt.subplots(nrows=len(plotting_columns), ncols=1)

    for ax, col in zip(axes, plotting_columns):
        # here we need to transpose the file name column scuh that the unique values
        # are now the the columns and the volues of `col` are the rows
        pivot_df = plotting_df.loc[:, ["filename", col]].pivot(index=col, columns="filename")
        nm_plots.plot_sabya(pivot_df, ax=ax, ax_ylabel=col)

    f.tight_layout()
    f.savefig(report_dir / "noboxplot_summary.png")


def distribution_comparison(
    model_output_dir: MODEL_OUT_DIR,
    label_dir: LABEL_DIR,
    report_dir: REPORT_DIR,
) -> None:
    """Generate a comparison of the results considered as a random distribution."""
    pass
