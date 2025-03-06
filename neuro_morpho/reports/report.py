"""Reports used for summarizing the results of the analysis."""

from pathlib import Path
from typing import Callable

import pandas as pd

ERR_NOT_IMPLEMENTED = "The {name} method is not implemented"

MODEL_OUT_DIR = str | Path
LABEL_DIR = str | Path
REPORT_FNAME = str | Path
report_fn = Callable[[MODEL_OUT_DIR, LABEL_DIR, REPORT_FNAME], None]


def _aggregate_results(
    report_dir: MODEL_OUT_DIR,
    output_dir: MODEL_OUT_DIR,
) -> pd.DataFrame:
    """Aggregate the results of the analysis."""
    pass


def noboxplot_summary(
    report_dir: MODEL_OUT_DIR,
    output_dir: MODEL_OUT_DIR,
    report_fname: REPORT_FNAME,
) -> None:
    """Generate a summary of the results using a noboxplot-esque plot."""
    pass


def distribution_comparison(
    report_dir: MODEL_OUT_DIR,
    output_dir: MODEL_OUT_DIR,
    report_fname: REPORT_FNAME,
) -> None:
    """Generate a comparison of the results considered as a random distribution."""
    pass
