"""Data Loader for NeuroMorpho Dataset."""

from pathlib import Path

import cv2
import gin
import torch
import torch.utils.data as td
from torchvision.transforms import v2


class NeuroMorphoDataset(td.Dataset):
    """NeuroMorpho Dataset."""

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
            data_dir (str|Path): Directory containing the data.
            transform (v2.Transform, optional): Transform to be applied to the data. Defaults to None.
        """
        self.img_files = list(Path(x_dir).glob("*.tif"))
        self.lbl_files = list(Path(y_dir).glob("*.tif"))
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
        data_dir (str|Path): Directory containing the data.
        batch_size (int, optional): Batch size. Defaults to 1.
        shuffle (bool, optional): Whether to shuffle the data. Defaults to True.
        num_workers (int, optional): Number of workers. Defaults to 0.
        transform (v2.Transform, optional): Transform to be applied to the data. Defaults to None.

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


if __name__ == "__main__":
    # Example usage
    from tqdm import tqdm

    import neuro_morpho.model.transforms as t

    x_dir = Path("data/new/imgs")
    y_dir = Path("data/new/lbls")
    print(len(list(x_dir.glob("*.tif"))), len(list(y_dir.glob("*.tif"))))
    dataloader = build_dataloader(
        x_dir,
        y_dir,
        batch_size=4,
        num_workers=4,
        aug_transform=v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(dtype=torch.float32),
                v2.RandomCrop(size=(1024, 1024)),  # Random crop for training images
            ]
        ),
        y_norm=v2.Compose(
            [
                t.Norm2One(),
                t.DownSample(in_size=(1024, 1024), factors=(1.0, 0.50, 0.25, 0.125)),
            ]
        ),
    )
    for i in tqdm(range(5), position=0, leave=True):
        for img, lbl in tqdm(dataloader, position=1, leave=True):
            pass
