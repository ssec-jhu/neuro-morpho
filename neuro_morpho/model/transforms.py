"""Transforms for the NeuroMorpho dataset."""

from typing import override

import gin
import torch


@gin.configurable(allowlist=["lbl_idx"])
class Standardize(torch.nn.Module):
    def __init__(self, lbl_idx: int):
        """Performs standardization on the input image channels.

        Args:
            lbl_idx (int): Index of the label channel to normalize against.
        """
        super().__init__()
        self.lbl_idx = lbl_idx

    @override
    def forward(self, stack: torch.Tensor) -> torch.Tensor:
        img, lbl = stack[: self.lbl_idx], stack[self.lbl_idx :]  # [n_lbls, h, w]

        return torch.stack(
            [
                (img - img.mean(dim=(1, 2, 3), keepdim=False)) / img.std(dim=(1, 2, 3), keepdim=False),
                lbl,
            ],
            dim=0,
        )


class Identity(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @override
    def forward(self, stack: torch.Tensor) -> torch.Tensor:
        """Forward pass for identity transform. Returns the input tensor unchanged.

        Args:
            stack (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: The same input tensor.
        """
        return stack
