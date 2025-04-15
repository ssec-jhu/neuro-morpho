import pytest

import neuro_morpho.logging.base as base


def test_base_logger_raises():
    """Test that the base logger raises NotImplementedError for all methods."""

    logger = base.Logger()

    with pytest.raises(NotImplementedError):
        logger.log_scalar("test_metric", 0.5, 1, True)

    with pytest.raises(NotImplementedError):
        logger.log_triplet(in_img=None, lbl_img=None, out_img=None, name="test_triplet", step=1, train=True)

    with pytest.raises(NotImplementedError):
        logger.log_parameters({"test_metric": 0.5})

    with pytest.raises(NotImplementedError):
        logger.log_code("test_folder")
