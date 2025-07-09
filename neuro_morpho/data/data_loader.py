"""Data Loader for the NeuroMorpho dataset."""

from pathlib import Path

import cv2
import gin
import torch
import torch.utils.data as td
from torchvision.transforms import v2


class NeuroMorphoDataset(td.Dataset):
    """NeuroMorpho Dataset.

    This dataset is used to load images and their corresponding labels for
    training and testing.
    """

    def __init__(
        self,
        x_dir: str | Path,
        y_dir: str | Path,
        aug_transform: v2.Transform = None,
        x_norm: v2.Transform = None,
        y_norm: v2.Transform = None,
    ):
        """Initialize the dataset.

        Args:
            x_dir (str|Path): Directory containing the input images.
            y_dir (str|Path): Directory containing the label images.
            aug_transform (v2.Transform, optional): Transform to be applied to
                the data for augmentation. Defaults to None.
            x_norm (v2.Transform, optional): Transform to be applied to the
                input images for normalization. Defaults to None.
            y_norm (v2.Transform, optional): Transform to be applied to the
                label images for normalization. Defaults to None.
        """
        self.img_files = [f for ext in ("*.pgm", "*.tif") for f in Path(x_dir).glob(ext)]
        self.lbl_files = [f for ext in ("*.pgm", "*.tif") for f in Path(y_dir).glob(ext)]
        self.img_files.sort()
        self.lbl_files.sort()

        self.aug_transform = aug_transform
        self.x_norm = x_norm
        self.y_norm = y_norm

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor | tuple[torch.Tensor, ...]]:
        """Get an item from the dataset.

        Args:
            index (int): Index of the item.

        Returns:
            tuple: Tuple containing the image and label.
        """
        img = self.img_files[index]
        lbl = self.lbl_files[index]

        img = cv2.imread(str(img), cv2.IMREAD_UNCHANGED)  # [h, w, 1]
        lbl = cv2.imread(str(lbl), cv2.IMREAD_GRAYSCALE)  # [h, w, n_lbls]

        stack = torch.cat(
            [
                torch.transpose(torch.atleast_3d(torch.from_numpy(img)), 0, 2).float(),
                torch.transpose(torch.atleast_3d(torch.from_numpy(lbl)), 0, 2).float(),
            ],
            dim=0,
        )  # [n_lbls+1, h, w]

        stack = self.aug_transform(stack) if self.aug_transform else stack

        img = stack[:1, ...]  # [1, h, w]
        lbl = stack[1:, ...]  # [n_lbls, h, w]

        return (img if self.x_norm is None else self.x_norm(img), lbl if self.y_norm is None else self.y_norm(lbl))

    def __len__(self) -> int:
        """Get the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.img_files)


@gin.configurable
def build_dataloader(
    x_dir: str | Path,
    y_dir: str | Path,
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 0,
    aug_transform: v2.Transform = None,
    x_norm: v2.Transform = None,
    y_norm: v2.Transform = None,
) -> td.DataLoader:
    """Build a DataLoader for the dataset.

    Args:
        x_dir (str|Path): Directory containing the input images.
        y_dir (str|Path): Directory containing the label images.
        batch_size (int, optional): Batch size. Defaults to 1.
        shuffle (bool, optional): Whether to shuffle the data. Defaults to True.
        num_workers (int, optional): Number of workers. Defaults to 0.
        aug_transform (v2.Transform, optional): Transform to be applied to the
            data for augmentation. Defaults to None.
        x_norm (v2.Transform, optional): Transform to be applied to the input
            images for normalization. Defaults to None.
        y_norm (v2.Transform, optional): Transform to be applied to the label
            images for normalization. Defaults to None.

    Returns:
        td.DataLoader: DataLoader for the dataset.
    """
    dataset = NeuroMorphoDataset(
        x_dir=x_dir,
        y_dir=y_dir,
        aug_transform=aug_transform,
        x_norm=x_norm,
        y_norm=y_norm,
    )

    return td.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
