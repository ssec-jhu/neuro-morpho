"""Transforms for the NeuroMorpho dataset."""

from typing import override

import gin
import torch
from torchvision.transforms import v2


@gin.register
class Standardize(torch.nn.Module):
    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x - x.mean(dim=(1, 2), keepdim=False) / x.std(dim=(1, 2), keepdim=False)


@gin.configurable(allowlist=["factors"])
class DownSample(torch.nn.Module):
    """Downsamples the input tensor by indicated factors."""

    def __init__(
        self,
        in_size: tuple[int, int],
        factors: int | tuple[float] | list[float],
    ):
        """Downsamples the input tensor by indicated factors.

        Args:
            factors (int | list[int]): Downsampling factors for each  w/h dimensions.
        """
        super().__init__()
        h, w = in_size

        if isinstance(factors, int | float):
            self.transforms = v2.Resize((int(h * factors), int(w * factors)))
        if isinstance(factors, tuple | list):
            self.transforms = tuple(v2.Resize((int(h * factor), int(w * factor))) for factor in factors)
        else:
            raise TypeError(f"Invalid type for factors: {type(factors)}")

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Downsample the input tensor.

        Args:
            stack (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Downsampled tensor.
        """

        return x


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
