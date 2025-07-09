# Adapted from:
# https://github.com/ssec-jhu/skeletonization/blob/master/solver/loss.py
# With the following license:
# MIT License

# Copyright (c) 2025 Nam Nguyen

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Loss functions for training models."""

from collections.abc import Callable

import gin
import torch
import torchvision

NAME_LOSS = tuple[str, torch.Tensor]
PRED = torch.Tensor
TARGET = torch.Tensor

LOSS_FN = Callable[[PRED, TARGET], tuple[NAME_LOSS, ...]]


@gin.configurable(allowlist=["alpha", "gamma", "reduction"])
class WeightedFocalLoss(torch.nn.Module):
    """Weighted version of Focal Loss.

    This loss is designed to address class imbalance by down-weighting easy
    examples and focusing on hard examples.

    See: https://arxiv.org/pdf/1708.02002

    Args:
        alpha (float): Weighting factor in range (0, 1) to balance positive vs negative examples.
        gamma (float): Focusing parameter to reduce the relative loss for well-classified examples.
        reduction (str): Specifies the reduction to apply to the output: 'none', 'mean', 'sum'.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2, reduction: str = "mean"):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> tuple[str, torch.Tensor]:
        """Calculate the weighted focal loss."""
        return "weighted_focal_loss", torchvision.ops.sigmoid_focal_loss(
            inputs,
            targets,
            alpha=self.alpha,
            gamma=self.gamma,
            reduction=self.reduction,
        )


@gin.configurable(allowlist=["smooth"])
class DiceLoss(torch.nn.Module):
    """Dice Loss for image segmentation.

    This loss is commonly used for image segmentation tasks. It measures the
    overlap between the predicted and target segmentations.
    """

    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(
        self,
        preds: torch.Tensor | list[torch.Tensor],
        targets: torch.Tensor | list[torch.Tensor],
    ) -> tuple[str, torch.Tensor]:
        """Calculate the dice loss."""
        numerator = 2 * torch.sum(preds * targets) + self.smooth
        denominator = torch.sum(preds**2) + torch.sum(targets**2) + self.smooth
        soft_dice_loss = 1 - numerator / denominator

        return "dice_loss", soft_dice_loss


@gin.configurable(allowlist=["loss_fn", "coefs"])
class WeightedMap(torch.nn.Module):
    """Weighted Map Loss.

    This loss applies a weighted sum of a given loss function to a list of
    predictions and targets.
    """

    def __init__(self, loss_fn: torch.nn.Module, coefs: list[float]):
        super(WeightedMap, self).__init__()
        self.coefs = coefs
        self.loss_fn = loss_fn

    def forward(self, pred: list[torch.Tensor], lbl: list[torch.Tensor]) -> tuple[str, torch.Tensor]:
        """Calculate the weighted map loss."""
        total_loss = 0
        for i in range(len(pred)):
            name, loss = self.loss_fn(pred[i], lbl[i])
            total_loss += loss * self.coefs[i]
        return name, total_loss


@gin.configurable(allowlist=["weights", "losses"])
class CombinedLoss(torch.nn.Module):
    """Combined Loss Function.

    This loss function combines multiple loss functions with given weights.
    """

    def __init__(self, weights: list[float], losses: list[torch.nn.Module]):
        """
        Args:
            weights: A tensor of weights for each loss function.
            losses: A list of loss functions to combine.
        """
        super(CombinedLoss, self).__init__()
        self.weights = weights
        self.losses = losses

    def forward(self, pred: torch.Tensor, lbl: torch.Tensor) -> list[tuple[str, torch.Tensor]]:
        """Forward pass to compute the combined loss.

        Args:
            pred: The predicted tensor.
            lbl: The target/label tensor.

        Returns:
            list[tuple[str, torch.Tensor]]: A list of tuples, where each tuple
                contains the name of the loss and the weighted loss value.
        """

        name_vals = [loss(pred, lbl) for loss in self.losses]
        return [(name, val * weight) for (name, val), weight in zip(name_vals, self.weights)]
