from pathlib import Path

from neuro_morpho.reports import report


def generate_report(
    pred_dir: str | Path,
    label_dir: str | Path,
    output_dir: str | Path,
) -> None:
    # use SKAN here to generate the report
    pass


def generate_plots(report_dir: str | Path, output_dir: str | Path, reports: list[report.report_fn]) -> None:
    for report_fn in reports:
        report_fn(report_dir, output_dir)
