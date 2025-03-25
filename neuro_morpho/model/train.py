from pathlib import Path

import neuro_morpho.model.base as base


def train_model(
    model: base.BaseModel,
    data_dir: str | Path,
) -> base.BaseModel:
    """Train the model on the given data directory.

    Args:
        model (BaseModel): The model to train
        data_dir (str|Path): The directory containing the data files to fit the model
            images should have the size (n_samples, width, height, channels)

    Returns:
        BaseModel: The trained model
    """
    return model.fit(data_dir)
