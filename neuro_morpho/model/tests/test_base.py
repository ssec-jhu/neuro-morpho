"""Test for the BaseModel"""

import pytest

from neuro_morpho.model.base import BaseModel


def test_base_model_init_raises() -> None:
    """Test that all base model methods raise"""

    model = BaseModel()

    with pytest.raises(NotImplementedError):
        model.fit("data_dir")

    with pytest.raises(NotImplementedError):
        model.predict_dir("in_dir", "out_dir")

    with pytest.raises(NotImplementedError):
        model.predict(None)

    with pytest.raises(NotImplementedError):
        model.predict_proba(None)

    with pytest.raises(NotImplementedError):
        model.save("path")

    with pytest.raises(NotImplementedError):
        model.load("path")
