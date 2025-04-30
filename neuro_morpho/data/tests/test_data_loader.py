from pathlib import Path

import numpy as np
import skimage as ski

import neuro_morpho.data.data_loader as dl


def setup_mock_data(tmp_path: str | Path) -> None:
    """Set up mock data for testing.

    Args:
        tmp_path (str|Path): Directory to store the mock data.
    """
    # Create mock data
    # This is a placeholder. You should create actual mock data files in the test_dir.
    x_dir = tmp_path / "img"
    y_dir = tmp_path / "lbl"
    x_dir.mkdir(parents=True, exist_ok=True)
    y_dir.mkdir(parents=True, exist_ok=True)

    for i in range(5):
        ski.io.imsave(x_dir / f"img_{i}.pgm", np.ones((256, 256)) * i, check_contrast=False)
        ski.io.imsave(y_dir / f"img_{i}.pgm", np.ones((256, 256)) * i, check_contrast=False)


def test_NeuroMorphoDataset(tmp_path: str | Path) -> None:
    """Test the NeuroMorphoDataset class.

    Args:
        tmp_path (str|Path): Directory to store the mock data.
    """
    setup_mock_data(tmp_path)

    ds = dl.NeuroMorphoDataset(
        x_dir=tmp_path / "img",
        y_dir=tmp_path / "lbl",
        aug_transform=None,
        x_norm=None,
        y_norm=None,
    )

    for i in range(len(ds)):
        img, lbl = ds[i]
        assert img.shape == (1, 256, 256), f"Expected image shape (1, 256, 256), got {img.shape}"
        assert lbl.shape == (1, 256, 256), f"Expected label shape (5, 256, 256), got {lbl.shape}"


def test_data_loader(tmp_path: Path) -> None:
    """Test the data loader.

    Args:
        tmp_path (Path): Temporary directory for testing.
    """
    # Set up mock data
    setup_mock_data(tmp_path)

    # Create a DataLoader
    dataloader = dl.build_dataloader(
        x_dir=tmp_path / "img",
        y_dir=tmp_path / "lbl",
        batch_size=1,
        shuffle=False,
        num_workers=0,
        aug_transform=None,
        x_norm=None,
        y_norm=None,
    )

    # Test the DataLoader
    for batch in dataloader:
        assert batch is not None
