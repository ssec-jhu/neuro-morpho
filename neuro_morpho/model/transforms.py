"""Transforms for the NeuroMorpho dataset."""

import gin
import torch
from torchvision.transforms import v2
from typing_extensions import override


@gin.register
class Standardize(torch.nn.Module):
    """Standardize the input tensor by subtracting the mean and dividing by the standard deviation."""

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - x.mean(dim=(1, 2), keepdim=False)) / (x.std(dim=(1, 2), keepdim=False) + self.eps)


@gin.register
class Norm2One(torch.nn.Module):
    """Normalize the input tensor to the range [0, 1]."""

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x / (x.max() + self.eps)  # Add small epsilon to avoid division by zero


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
