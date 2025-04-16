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


@gin.configurable(allowlist=["alpha", "gamma", "coefs"])
class WeightedFocalLoss(torch.nn.Module):
    "Non weighted version of Focal Loss"

    def __init__(self, beta=0.25, gamma=2, coefs: list[float] = None):
        super(WeightedFocalLoss, self).__init__()
        self.beta = torch.tensor(beta).cuda()
        self.gamma = gamma
        self.coefs = coefs

    def forward(self, inputs, targets) -> tuple[str, torch.Tensor]:
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        targets = targets.type(torch.long)
        pt = torch.exp(-BCE_loss)
        F_loss = self.beta * (1 - pt) ** self.gamma * BCE_loss
        return "weighted_focal_loss", F_loss.mean()


@gin.configurable(allowlist=["smooth"])
class DiceLoss(torch.nn.Module):
    """Dice Loss that can handle multiscale outputs."""

    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(
        self,
        preds: torch.Tensor | list[torch.Tensor],
        targets: torch.Tensor | list[torch.Tensor],
    ) -> tuple[str, torch.Tensor]:
        numerator = 2 * torch.sum(preds * targets) + self.smooth
        denominator = torch.sum(preds**2) + torch.sum(targets**2) + self.smooth
        soft_dice_loss = 1 - numerator / denominator

        return "dice_loss", soft_dice_loss


@gin.configurable(allowlist=["loss_fn", "coefs"])
class WeightedMap(torch.nn.Module):
    """Weighted Map Loss."""

    def __init__(self, loss_fn: torch.nn.Module, coefs: list[float]):
        super(WeightedMap, self).__init__()
        self.coefs = coefs
        self.loss_fn = loss_fn

    def forward(self, pred: list[torch.Tensor], lbl: list[torch.Tensor]) -> tuple[str, torch.Tensor]:
        total_loss = 0
        for i in range(len(pred)):
            name, loss = self.loss_fn(pred[i], lbl[i])
            total_loss += loss * self.coefs[i]
        return name, total_loss


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
        """Forward pass to compute the combined loss.

        Args:
            pred: The predicted tensor.
            lbl: The target/label tensor.

        Returns:
            tuple[str, torch.Tensor]: The losses weighted by weights.
        """

        name_vals = [loss(pred, lbl) for loss in self.losses]
        return [(name, val * weight) for (name, val), weight in zip(name_vals, self.weights)]
