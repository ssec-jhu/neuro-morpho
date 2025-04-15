"""Test cases for the TextLogger class in the neuro_morpho.logging.text module."""

from pathlib import Path

from neuro_morpho.logging.text import TextLogger


def test_text_logger_init(tmp_path: Path) -> None:
    """Test that the TextLogger initializes correctly."""
    TextLogger(tmp_path)

    assert (tmp_path / "triplets").exists()
    assert (tmp_path / "train").exists()
    assert (tmp_path / "test").exists()
