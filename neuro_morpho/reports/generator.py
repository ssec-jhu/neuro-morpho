"""Generate reports for the NeuroMorpho dataset."""

import json
from collections import defaultdict
from pathlib import Path

import gin
import pandas as pd
import skimage as ski

import neuro_morpho.reports.report as nm_reports
import neuro_morpho.reports.stats as nm_stats


def _aggregate_results(
    stats_dir: str | Path,
) -> pd.DataFrame:
    """Aggregate the results of the analysis.

    The results of the ananlysis is returned as a pandas DataFrame. It is also
    saved in the `stats_dir` directory as a csv file named `aggregated_results.csv`.

    Args:
        stats_dir (str|Path): The directory containing the statistics files.

    Returns:
        pd.DataFrame: The aggregated results.
    """
    stat_files = Path(stats_dir).glob("*.json")

    aggregated = defaultdict(list)
    for stat_file in stat_files:
        with stat_file.open() as f:
            stats = json.load(f)
            for k, v in stats.items():
                aggregated[k].append(v)

    aggregated_df = pd.DataFrame(aggregated)
    aggregated_df.to_csv(stats_dir / "aggregated_results.csv")

    return aggregated_df


def _parse_single_file(
    input_file: str | Path,
    output_file: str | Path,
) -> None:
    skeleton_stats = {
        "fname": input_file,
        **nm_stats.skeleton_analysis(
            ski.io.imread(input_file),
        ),
    }

    with open(output_file, "w") as f:
        json.dump(skeleton_stats, f)


def generate_statistics(
    in_dir: str | Path,
    out_dir: str | Path,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    in_pairs = list(
        map(
            lambda x: (x, out_dir / x.name),
            Path(in_dir).glob("*.pgm"),
        )
    )

    for in_file, out_file in in_pairs:
        _parse_single_file(in_file, out_file)

    _aggregate_results(out_dir)


@gin.configurable(allowlist=["reports"])
def generate_report(
    model_out_dir: str | Path,
    labeled_out_dir: str | Path,
    report_out_path: str | Path,
    reports: list[nm_reports.report_fn],
) -> None:
    report_out_path = Path(report_out_path)
    report_out_path.mkdir(parents=True, exist_ok=True)

    for report_fn in reports:
        report_fn(model_out_dir, labled_out_dir, report_out_path)
