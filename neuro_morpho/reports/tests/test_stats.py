"""Test for the stats module."""

from functools import partial

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
    ("include_isolated_branches", "expected_n_tip_points"),
    [
        (False, 2),  # Default case: 1 junction-to-endpoint + 2 junction-to-junction
        (True, 3),  # Include isolated branches: adds 2 isolated branches
    ],
)
def test_calculate_n_tip_points(
    include_isolated_branches: bool,
    expected_n_tip_points: int,
) -> None:
    """Tests the calculate_n_tip_points function."""
    df = generate_sample_data()

    actual_n_tip_points = stats.calculate_n_tip_points(df, include_isolated_branches=include_isolated_branches)
    assert actual_n_tip_points == expected_n_tip_points


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


def test_calculate_total_length_raises() -> None:
    """Tests that calculate_total_length raises an error for unsupported distance types."""
    df = generate_sample_data()

    with pytest.raises(ValueError, match=stats.ERR_INVALID_DIST):
        stats.calculate_total_length(df, dist_type="unsupported")


def test_calculate_branch_lengths_raises() -> None:
    """Tests that calculate_total_length raises an error for unsupported distance types."""
    df = generate_sample_data()

    with pytest.raises(ValueError, match=stats.ERR_INVALID_DIST):
        stats.calculate_branch_lengths(df, dist_type="unsupported")


@pytest.mark.parametrize(
    ("assume_single_skeleton",),
    [(True,), (False,)],
)
def test_skeleton_analysis(
    assume_single_skeleton: bool,
) -> None:
    """Tests the skeleton analysis function."""

    result = stats.skeleton_analysis(
        generate_sample_skeleton(),
        [
            (
                "n_branches",
                partial(stats.calculate_n_branches, include_isolated_branches=True, include_isolated_cycles=True),
            ),
        ],
        assume_single_skeleton=assume_single_skeleton,
    )

    n_branches = [4] if assume_single_skeleton else [1, 3]
    assert isinstance(result, dict)
    for sub_id, n_branches in zip(result, n_branches, strict=True):
        assert isinstance(result[sub_id], dict)
        assert "n_branches" in result[sub_id]
        assert result[sub_id]["n_branches"] == n_branches, (
            f"Expected {n_branches} branches, got {result[sub_id]['n_branches']} for sub_id {sub_id}"
        )
