"""Tests for the report generator module."""

import json
from pathlib import Path

import gin
import numpy as np
import skimage as ski

from neuro_morpho.reports import generator


def generate_sample_skeleton() -> np.ndarray:
    """Generates a sample skeleton for testing."""
    # This is a simple 2D skeleton represented as a binary image
    # where 1s represent the skeleton and 0s represent the background.
    return np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
        ]
    )


def test_aggregate_results(tmp_path: Path):
    """Test the aggregation of results."""
    # Create a temporary directory
    stats_dir = tmp_path / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)

    # Create some test JSON files
    test_data = [
        {"filename": "test1.json", 1: {"stat1": 0.5, "stat2": 0.7}},
        {"filename": "test2.json", 1: {"stat1": 0.6, "stat2": 0.8}},
    ]

    for data in test_data:
        with open(stats_dir / f"{data['filename']}", "w") as f:
            json.dump(data, f)

    # Call the function to aggregate results
    aggregated_df = generator._aggregate_results(stats_dir)

    # Check if the aggregated DataFrame is correct
    assert len(aggregated_df) == len(test_data)
    assert set(aggregated_df.columns) == {"filename", "stat1", "stat2"}


def test_parse_single_file(tmp_path: Path):
    """Test the parsing of a single file."""
    # Create a temporary directory
    input_file = tmp_path / "input.pgm"
    output_file = tmp_path / "output.json"
    expected_outs = {
        "filename": str(input_file),
        "0": {
            "stat1": 0.5,
        },
        "1": {
            "stat1": 0.5,
        },
    }

    ski.io.imsave(input_file, generate_sample_skeleton().astype(np.uint8))

    # Call the function to parse the single file
    gin.clear_config()
    gin.bind_parameter(
        "neuro_morpho.reports.stats.skeleton_analysis.stat_fns",
        [
            ("stat1", lambda _: 0.5),
        ],
    )
    generator._parse_single_file(input_file, output_file)

    # Check if the output file is correct
    with open(output_file, "r") as f:
        output_data = json.load(f)
        assert output_data == expected_outs


def test_generate_statistics(tmp_path: Path):
    """Test the generation of statistics."""
    # Create a temporary directory
    in_dir = tmp_path / "input"
    out_dir = tmp_path / "output"
    in_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Create some test input files
    for i in range(3):
        input_file = in_dir / f"test_{i}.pgm"
        ski.io.imsave(input_file, generate_sample_skeleton().astype(np.uint8))

    # Call the function to generate statistics
    gin.clear_config()
    gin.bind_parameter(
        "neuro_morpho.reports.stats.skeleton_analysis.stat_fns",
        [
            ("stat1", lambda _: 0.5),
        ],
    )
    generator.generate_statistics(in_dir, out_dir)

    # Check if the output files are created
    for i in range(3):
        assert (out_dir / f"test_{i}.json").exists()
