"""Loss functions for neuro_morpho."""

from collections.abc import Callable

import torch

NAME_LOSS = tuple[str, torch.Tensor]
PRED = torch.Tensor
TARGET = torch.Tensor

LOSS_FN = Callable[[PRED, TARGET], tuple[NAME_LOSS, ...]]
