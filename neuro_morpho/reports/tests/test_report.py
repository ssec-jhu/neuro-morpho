"""Test the report generation functionality."""

from pathlib import Path

from neuro_morpho.reports import report


def test_transform_filename_to_group() -> None:
    """Test the transform_filename_to_group function."""

    assert report.transform_filename_to_group("test_file") == "Label"
    assert report.transform_filename_to_group("test_file_sbr-1") == "Model(1)"
    assert report.transform_filename_to_group("test_file_SBR-2") == "Model(2)"
    assert report.transform_filename_to_group("test_file_sbr-5") == "Model(5)"


def test_noboxplot_summary(tmp_path: Path) -> None:
    """Test the noboxplot_summary function."""
