"""Generates plots and reports for model comparison."""

from typing import Callable

import numpy as np
import pandas as pd
import skan
import skimage as ski

VALID_DISTANCES = {"euclidean", "manhattan"}
ERR_INVALID_DIST = f"Invalid distance type. Must be one of {VALID_DISTANCES}"

SKELETON_STAT_FN = Callable[[pd.DataFrame], np.ndarray]


def extract_branch_ids(skan_skel_data: pd.DataFrame) -> set[int]:
    """Extract the branch ids from the skan skeleton data.

    Args:
        skan_skel_data (pd.DataFrame): The skan skeleton data.

    Returns:

    """
    # branch ids are those ids that appear at least twice in the list of branches
    # otherwise they are endpoints
    ids, cnts = np.unique(
        skan_skel_data.loc[:, ["node_id_src", "node_id_dst"]].values.flatten(),
        return_counts=True,
    )
    return set(ids[cnts > 1])


def calculate_n_branches(
    skan_skel_data: pd.DataFrame,
    *,
    include_isolated_branches: bool = False,
    include_isolated_cycles: bool = False,
) -> int:
    """Calculate the number of branches in the skeleton data.

    Args:
        skan_skel_data (pd.DataFrame): The skan skeleton data.

    Returns:
        The number of branches in the skeleton data.
    """
    # Always include junction-to-junction and junction-to-endpoint branches
    # optionally include isolated branches and isolated cycles
    types_to_include = [1, 2] + ([0] * include_isolated_branches) + ([3] * include_isolated_cycles)

    return len(skan_skel_data[skan_skel_data["branch_type"].isin(set(types_to_include))])


def calculate_n_tip_points(
    skan_skel_data: pd.DataFrame,
    *,
    include_isolated_branches: bool = False,
) -> int:
    """Calculate the number of tip points in the skeleton data.

    Args:
        skan_skel_data (pd.DataFrame): The skan skeleton data.

    Returns:
        The number of tip points in the skeleton data.
    """
    # Always include endpoint-to-endpoint branches
    # optionally include isolated branches
    types_to_include = [1] + ([0] * include_isolated_branches)

    return len(skan_skel_data[skan_skel_data["branch_type"].isin(set(types_to_include))])


def calculate_total_length(
    skan_skel_data: pd.DataFrame,
    dist_type: str = "euclidean",
) -> float:
    """Calculate the total length of the skeleton data.

    Args:
        skan_skel_data (pd.DataFrame): The skan skeleton data.
        dist_type (str): The type of distance to use for the length calculation.

    Returns:
        The total length of the skeleton data.
    """
    if dist_type not in VALID_DISTANCES:
        raise ValueError(ERR_INVALID_DIST)

    return calculate_branch_lengths(skan_skel_data, dist_type).sum()


def calculate_branch_lengths(skan_skel_data: pd.DataFrame, dist_type: str = "euclidean") -> np.ndarray:
    """Calculate the lengths of each branch in the skeleton data.

    Args:
        skan_skel_data (pd.DataFrame): The skan skeleton data.
        dist_type (str): The type of distance to use for the length calculation.

    Returns:
        a numpy array of the branch lengths
    """
    if dist_type not in VALID_DISTANCES:
        raise ValueError(ERR_INVALID_DIST)

    distances = np.array([0] * len(skan_skel_data))
    if dist_type == "euclidean":
        distances = skan_skel_data["euclidean_distance"].values
    else:
        dim0 = (skan_skel_data["coord_src_0"] - skan_skel_data["coord_dst_0"]).abs()
        dim1 = (skan_skel_data["coord_src_1"] - skan_skel_data["coord_dst_1"]).abs()
        distances = (dim0 + dim1).values

    return distances


def skeleton_analysis(
    skeleton: np.ndarray,
    stat_fns: tuple[list[str], list[SKELETON_STAT_FN]],
    pixel_size: float = 1,
    *,
    assume_single_skeleton: bool = False,
):
    """Generate a summary of the skeleton analysis.

    Args:
        skeleton (np.ndarray): The skeleton of the image to analyze, should be 2d.
        stat_fns (tuple[list[str], list[SKELETON_STAT_FN]]): The list of functions to
            use for the analysis.
        pixel_size (float): The size of the pixel in the image.
    """
    skeleton = skan.Skeleton(skeleton, spacing=pixel_size)
    # branch_data is a pandas DataFrame
    # branch_type can be one of the following:
    # 0 = endpoint-to-endpoint (isolated branch)
    # 1 = junction-to-endpoint
    # 2 = junction-to-junction
    # 3 = isolated cycle
    branch_data = skan.summarize(skeleton, separator="_").loc[
        :,
        [
            "skeleton_id",  # each sub skeletong in an image has a unique value
            "branch_type",  # indictates the type of branch, see above comment
            "node_id_src",  # the source node id, src_id is always less than dst_id, except for maybe cycles?
            "node_id_dst",  # the destination node id
            "euclidean_distance",  # euclidean distance between src and dst
            "coord_src_0",  # y coordinate of src, in the units of pixel*pixel_size
            "coord_src_1",  # x coordinate of src, in the units of pixel*pixel_size
            "coord_dst_0",  # y coordinate of dst, in the units of pixel*pixel_size
            "coord_dst_1",  # x coordinate of dst, in the units of pixel*pixel_size
        ],
    ]

    # calculate the statistics
    grouped_data = [(1, branch_data)] if assume_single_skeleton else branch_data.groupby("skeleton_id")

    stats = {}
    for skeleton_id, sub_df in grouped_data:
        stats[skeleton_id] = {}
        for stat_name, stat_fn in stat_fns:
            stats[skeleton_id][stat_name] = stat_fn(sub_df)

    return stats


if __name__ == "__main__":
    import functools

    skeleton = ski.io.imread("/home/ryanhausen/Downloads/Skeleton-Sample-2-time-100.00.pgm")
    stat_fns = [
        ("n_branches", calculate_n_branches),
        ("n_tip_points", calculate_n_tip_points),
        ("total_length", functools.partial(calculate_total_length, dist_type="euclidean")),
        ("branch_lengths", functools.partial(calculate_branch_lengths, dist_type="euclidean")),
    ]

    branch_data = skeleton_analysis(skeleton, stat_fns, assume_single_skeleton=True)

    print(branch_data)
