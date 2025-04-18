"""Test for the stats module."""

import numpy as np
import pandas as pd
import pytest

from neuro_morpho.reports import stats


def generate_sample_data() -> pd.DataFrame:
    """Generates a sample DataFrame for testing."""
    data = {
        "node_id_src": [0, 0, 1, 2, 2],
        "node_id_dst": [1, 2, 3, 4, 5],
        "branch_type": [
            1,
            2,
            1,
            0,
            3,
        ],  # 1: junction-to-endpoint, 2: junction-to-junction, 0: isolated branch, 3: cycle
        "euclidean_distance": [
            1.0,
            2.0,
            2.0,
            2.0,
            np.sqrt((2 - 5) ** 2 + (2 - 5) ** 2),
        ],  # Example lengths for each branch
        "coord_src_0": [0, 0, 0, 0, 0],
        "coord_src_1": [0, 0, 1, 2, 2],
        "coord_dst_0": [0, 0, 0, 0, 5],
        "coord_dst_1": [1, 2, 3, 4, 5],
    }
    return pd.DataFrame(data)


def test_extract_branch_ids() -> None:
    """Tests the extract_branch_ids function."""
    df = generate_sample_data()

    expected_ids = {0, 1, 2}
    actual_branch_ids = stats.extract_branch_ids(df)
    assert actual_branch_ids == expected_ids


@pytest.mark.parametrize(
    ("include_isolated_branches", "include_isolated_cycles", "expected_n_branches"),
    [
        (False, False, 3),  # Default case
        (True, False, 4),  # Include isolated branches
        (False, True, 4),  # Include isolated cycles
        (True, True, 5),  # Include both isolated branches and cycles
    ],
)
def test_calcualte_n_branches(
    include_isolated_branches: bool,
    include_isolated_cycles: bool,
    expected_n_branches: int,
) -> None:
    """Test the calculate_n_branches function."""
    df = generate_sample_data()

    actual_n_branches = stats.calculate_n_branches(
        df, include_isolated_branches=include_isolated_branches, include_isolated_cycles=include_isolated_cycles
    )
    assert actual_n_branches == expected_n_branches


@pytest.mark.parametrize(
    ("dist_type", "expected_lengths"),
    [
        ("euclidean", np.array([1.0, 2.0, 2.0, 2.0, np.sqrt((2 - 5) ** 2 + (2 - 5) ** 2)])),
        ("manhattan", np.array([1.0, 2.0, 2.0, 2.0, 8.0])),
    ],
)
def test_calculate_branch_lengths(dist_type: str, expected_lengths: float) -> None:
    """Tests the calculate_branch_lengths function."""
    df = generate_sample_data()
    # Assuming a simple distance calculation for testing

    branch_lengths = stats.calculate_branch_lengths(df, dist_type=dist_type)
    np.testing.assert_equal(branch_lengths, expected_lengths)


@pytest.mark.parametrize(
    ("dist_type", "expected_length"),
    [
        ("euclidean", np.array([1.0, 2.0, 2.0, 2.0, np.sqrt((2 - 5) ** 2 + (2 - 5) ** 2)]).sum()),
        ("manhattan", np.array([1.0, 2.0, 2.0, 2.0, 8.0]).sum()),
    ],
)
def test_calculate_total_length(dist_type: str, expected_length: float) -> None:
    """Tests the calculate_total_length function."""
    df = generate_sample_data()
    # Assuming a simple distance calculation for testing

    total_length = stats.calculate_total_length(df, dist_type=dist_type)
    assert total_length == expected_length


if __name__ == "__main__":
    test_calcualte_n_branches()
