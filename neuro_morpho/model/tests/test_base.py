"""Test for the BaseModel"""

import pytest

from neuro_morpho.model.base import BaseModel


def test_base_model_init_raises() -> None:
    """Test that all base model methods raise"""

    model = BaseModel()

    with pytest.raises(NotImplementedError):
        model.fit("data_dir")

    with pytest.raises(NotImplementedError):
        model.predict_dir("in_dir", "out_dir", 0.5, "test", (512, 512), "nn", False, False)

    with pytest.raises(NotImplementedError):
        model.predict(None)

    with pytest.raises(NotImplementedError):
        model.predict_proba(None, None)

    with pytest.raises(NotImplementedError):
        model.find_threshold(None, None, None, "model_out_val_y_dir", 0.1, 0.9, 0.01)

    with pytest.raises(NotImplementedError):
        model.save("path")

    with pytest.raises(NotImplementedError):
        model.load("path")
