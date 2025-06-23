"""Test CometLogger class."""

import numpy as np

from neuro_morpho.logging.comet import CometLogger


def test_log_scalar():
    """Test logging of scalar values with CometLogger."""
    logger = CometLogger(
        api_key=None,
        project_name="test_project",
        workspace="test_workspace",
        auto_param_logging=True,
        auto_metric_logging=True,
        disabled=True,
    )

    logger.log_scalar(
        name="test_metric",
        value=0.5,
        step=1,
        train=True,
    )
    assert logger.experiment is not None
    assert logger.experiment.metrics["test_metric"] == 0.5


def test_log_triplet():
    """Test logging a triplet with the CometLogger."""
    logger = CometLogger(
        api_key=None,
        project_name="test_project",
        workspace="test_workspace",
        auto_param_logging=True,
        auto_metric_logging=True,
        disabled=True,
    )

    in_img = (np.eye(10, 10) + 1) * 5
    lbl_img = np.eye(10, 10) + 1
    out_img = (np.eye(10, 10) + 1) * 0.5

    # this currently doesn't do anything because comet experiments that disable
    # logging don't do anything with figures
    logger.log_triplet(
        in_img=in_img,
        lbl_img=lbl_img,
        out_img=out_img,
        name="test_triplet",
        step=1,
        train=True,
    )

    assert logger.experiment is not None


def test_log_parameters():
    """Test logging parameters with the CometLogger."""
    logger = CometLogger(
        api_key=None,
        project_name="test_project",
        workspace="test_workspace",
        auto_param_logging=True,
        auto_metric_logging=True,
        disabled=True,
    )

    metrics = {"learning_rate": 0.001, "batch_size": 32, "curr_step": 1}
    logger.log_parameters(metrics)

    assert logger.experiment is not None
    assert logger.experiment.params == metrics


def test_log_code():
    """Test logging code with the CometLogger."""
    logger = CometLogger(
        api_key=None,
        project_name="test_project",
        workspace="test_workspace",
        auto_param_logging=True,
        auto_metric_logging=True,
        disabled=True,
    )

    # this currently doesn't do anything because comet experiments that disable
    # logging don't do anything with code
    logger.log_code(folder=".")
    assert logger.experiment is not None
