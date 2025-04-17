"""Transforms for the NeuroMorpho dataset."""

from typing import override

import gin
import torch
from torchvision.transforms import v2


@gin.register
class Standardize(torch.nn.Module):
    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - x.mean(dim=(1, 2), keepdim=False)) / x.std(dim=(1, 2), keepdim=False)


@gin.register
class Norm2One(torch.nn.Module):
    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x / x.max()


@gin.configurable(allowlist=["in_size", "factors"])
class DownSample(torch.nn.Module):
    """Downsamples the input tensor by indicated factors."""

    def __init__(
        self,
        in_size: tuple[int, int],
        factors: int | float | tuple[float, ...] | list[float],
    ):
        """Downsamples the input tensor by indicated factors.

        Args:
            factors (int | list[int]): Downsampling factors for each  w/h dimensions.
        """
        super().__init__()
        h, w = in_size

        self._single_factor = isinstance(factors, int | float)

        def down_f(factor: tuple[float, float]) -> v2.Transform:
            return v2.Resize((int(h * factor), int(w * factor)), interpolation=v2.InterpolationMode.NEAREST)

        if self._single_factor:
            self.transforms = down_f(factors)
        else:
            self.transforms = tuple(down_f(factor) for factor in factors)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Downsample the input tensor.

        Args:
            stack (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Downsampled tensor.
        """
        if self._single_factor:
            return self.transforms(x)
        else:
            return tuple(t(x) for t in self.transforms)
