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

"""Loss functions for neuro_morpho."""

from collections.abc import Callable

import gin
import torch
import torch.nn.functional as F

NAME_LOSS = tuple[str, torch.Tensor]
PRED = torch.Tensor
TARGET = torch.Tensor

LOSS_FN = Callable[[PRED, TARGET], tuple[NAME_LOSS, ...]]


@gin.configurable(allowlist=["alpha", "gamma"])
class WeightedFocalLoss(torch.nn.Module):
    "Non weighted version of Focal Loss"

    def __init__(self, alpha=0.25, gamma=2):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1 - alpha]).cuda()
        self.gamma = gamma

    def forward(self, inputs, targets) -> tuple[str, torch.Tensor]:
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        targets = targets.type(torch.long)
        at = self.alpha.gather(0, targets.data.view(-1))
        pt = torch.exp(-BCE_loss)
        F_loss = at * (1 - pt) ** self.gamma * BCE_loss
        return "weighted_focal_loss", F_loss.mean()


@gin.configurable(allowlist=["smooth"])
class DiceLoss(torch.nn.Module):
    """Dice Loss"""

    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> tuple[str, torch.Tensor]:
        numerator = 2 * torch.sum(preds * targets) + self.smooth
        denominator = torch.sum(preds**2) + torch.sum(targets**2) + self.smooth
        soft_dice_loss = 1 - numerator / denominator

        return "dice_loss", soft_dice_loss


@gin.configurable(allowlist=["weights", "losses"])
class CombinedLoss(torch.nn.Module):
    """Combined Loss Function."""

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
        """
        Forward pass to compute the combined loss.

        Args:
            pred: The predicted tensor.
            lbl: The target/label tensor.

        Returns:
            tuple[str, torch.Tensor]: The losses weighted by weights.
        """

        name_vals = [loss(pred, lbl) for loss, (pred, lbl) in self.losses]
        return [(name, val * weight) for (name, val), weight in zip(name_vals, self.weights)]
