# Adapted from:
# https://github.com/namdvt/skeletonization/blob/master/model/unet_att.py
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
import functools
import itertools
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any, override

import gin
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as td
from tqdm import tqdm

import neuro_morpho.data.data_loader as data_loader
import neuro_morpho.logging.base as base_logging
from neuro_morpho.model import base, loss, metrics

ERR_PREDICT_DIR_NOT_IMPLEMENTED = (
    "The predict_dir method is not implemented, because you might be tiling, subclass and implement this method."
)


def apply_tpl(fn: Callable, item: Any | tuple[Any, ...]) -> Any | tuple:
    """Apply a function to a an item or to all of the items in a tuple."""
    return tuple(map(fn, item)) if isinstance(item, tuple) else fn(item)


def cast_and_move(tensor: torch.Tensor, device: str) -> torch.Tensor:
    """Cast and move tensor to the specified device."""
    return tensor.float().to(device)


def detach_and_move(tensor: torch.Tensor, idx: int | None = None) -> np.ndarray:
    """Detach and move tensor to the specified device."""
    if idx is None:
        return tensor.detach().cpu().numpy()
    else:
        return tensor[idx].detach().cpu().numpy()


@gin.register
class UNet(base.BaseModel):
    def __init__(
        self,
        n_input_channels: int = 1,
        n_output_channels: int = 1,
        encoder_channels: list[int] = [64, 128, 256, 512, 1024],
        decoder_channels: list[int] = [512, 256, 128, 64],
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super(UNet, self).__init__()
        self.model = UNetModule(
            n_input_channels=n_input_channels,
            n_output_channels=n_output_channels,
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
        ).to(device)
        self.cast_fn = functools.partial(cast_and_move, device=device)
        self.device = device

    @override
    def predict_dir(self, in_dir: Path | str, out_dir: Path | str):
        raise NotImplementedError(ERR_PREDICT_DIR_NOT_IMPLEMENTED)

    @gin.register(
        allowlist=[
            "train_data_loader",
            "test_data_loader",
            "epochs",
            "optimizer",
            "loss_fn",
            "metric_fns",
            "logger",
            "log_every",
        ]
    )
    def fit(
        self,
        training_x_dir: str | Path | None = None,
        training_y_dir: str | Path | None = None,
        testing_x_dir: str | Path | None = None,
        testing_y_dir: str | Path | None = None,
        train_data_loader: td.DataLoader = None,
        test_data_loader: td.DataLoader = None,
        epochs: int = 1,
        optimizer: torch.optim.Optimizer = None,
        loss_fn: loss.LOSS_FN = None,
        metric_fns: list[metrics.METRIC_FN] | None = None,
        logger: base_logging.Logger = None,
        log_every: int = 10,
        init_step: int = 0,
    ) -> base.BaseModel:
        if train_data_loader is None:
            train_data_loader = data_loader.build_dataloader(training_x_dir, training_y_dir)
        if test_data_loader is None:
            test_data_loader = data_loader.build_dataloader(testing_x_dir, testing_y_dir)

        optimizer = optimizer(params=self.model.parameters())

        step = init_step
        # TODO: steps needs to be fixed
        for n_epoch in tqdm(range(epochs), desc="Epochs", unit="epoch", position=0):
            self.model.train()
            for x, y in tqdm(train_data_loader, desc="Training", unit="batch", position=1):
                optimizer.zero_grad()

                x = self.cast_fn(x)
                y = self.cast_fn(y) if not isinstance(y, tuple | list) else tuple(map(self.cast_fn, y))

                pred = self.model(x)
                losses = loss_fn(pred, y)
                loss = sum(map(lambda lss: lss[1], losses)) if isinstance(losses, (tuple, list)) else losses[1]
                loss.backward()
                optimizer.step()

                if logger is not None and step % log_every == 0:
                    x = detach_and_move(x, idx=0 if isinstance(x, tuple | list) else None)
                    pred = detach_and_move(pred, idx=0 if isinstance(pred, tuple | list) else None)
                    y = detach_and_move(y, idx=0 if isinstance(y, tuple | list) else None)

                    fns_args = zip(metric_fns, itertools.repeat((pred, y), len(metric_fns)), strict=True)
                    metrics_values = [fn(pred, y) for fn, (pred, y) in fns_args]
                    for name, value in metrics_values:
                        logger.log_scalar(name, value, step=step, train=True)
                    for name, loss in losses:
                        logger.log_scalar(name, loss.item(), step=step, train=True)
                    logger.log_scalar("loss", loss.item(), step=step, train=True)

                    sample_idx = np.random.choice(x.shape[0], size=1)[0]
                    sample_x = x[sample_idx, ...].squeeze()
                    sample_y = y[sample_idx, ...].squeeze()
                    sample_pred = pred[sample_idx, ...].squeeze()

                    logger.log_triplet(sample_x, sample_y, sample_pred, "triplet", step=step, train=True)
                step += 1

            if logger is not None:
                self.model.eval()
                loss_numerator = defaultdict(float)
                loss_denominator = defaultdict(float)
                for x, y in tqdm(test_data_loader, desc="Testing", unit="batch", position=2):
                    with torch.no_grad():
                        x = apply_tpl(self.cast_fn, x)
                        y = self.cast_fn(y) if not isinstance(y, tuple | list) else tuple(map(self.cast_fn, y))

                        pred = self.model(x)
                        losses = loss_fn(pred, y)
                        loss = sum(losses) if isinstance(losses, tuple) else losses

                        for name, loss in losses:
                            loss_numerator[name] += loss.item()
                            loss_denominator[name] += x.shape[0]

                        x = detach_and_move(x, idx=0 if isinstance(x, tuple | list) else None)
                        pred = detach_and_move(pred, idx=0 if isinstance(pred, tuple | list) else None)
                        y = detach_and_move(y, idx=0 if isinstance(y, tuple | list) else None)

                        fns_args = zip(metric_fns, itertools.repeat((pred, y), len(metric_fns)), strict=True)
                        metrics_values = [fn(pred, y) for fn, (pred, y) in fns_args]
                        for name, value in metrics_values:
                            loss_numerator[name] += value * x.shape[0]  # accumulate total metric value
                            loss_denominator[name] += x.shape[0]

                        loss_numerator["loss"] += loss.item() * x.shape[0]  # accumulate total loss
                        loss_denominator["loss"] += x.shape[0]

                for name, num in loss_numerator.items():
                    logger.log_scalar(name, num / loss_denominator[name], step=step, train=False)

                sample_idx = np.random.choice(x.shape[0], size=1)[0]
                sample_x = x[sample_idx, ...].squeeze()
                sample_y = y[sample_idx, ...].squeeze()
                sample_pred = pred[sample_idx, ...].squeeze()
                logger.log_triplet(sample_x, sample_y, sample_pred, "triplet", step=step, train=False)

        return self

    @override
    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self.model(torch.from_numpy(x).float().to(self.device)).squeeze(1).cpu().detach().numpy()

    @override
    def save(self, path: str | Path) -> None:
        torch.save(self.model.state_dict(), path)

    @override
    def load(self, path: str | Path) -> None:
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()


class UNetModule(nn.Module):
    def __init__(
        self,
        n_input_channels: int = 1,
        n_output_channels: int = 1,
        encoder_channels: list[int] = [64, 128, 256, 512, 1024],
        decoder_channels: list[int] = [512, 256, 128, 64],
    ):
        super(UNetModule, self).__init__()
        self.encoder = Encoder(in_channels=n_input_channels, channels=encoder_channels)
        self.decoder = Decoder(
            in_channels=encoder_channels[-1],
            channels=decoder_channels,
            out_channels=n_output_channels,
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.encoder(x)
        return self.decoder(x)


class Encoder(nn.Module):
    def __init__(self, in_channels: int, channels: list[int], kernel_size: int = 3, padding: int = 1):
        super(Encoder, self).__init__()
        self._channels = channels
        channel_list = [in_channels] + channels

        for i, (in_ch, out_ch) in enumerate(itertools.pairwise(channel_list), start=1):
            setattr(self, f"conv{i}", DoubleConv2d(in_ch, out_ch, kernel_size, padding=padding))
            setattr(self, f"att{i}", AttentionGroup(out_ch))
        self.pooling = nn.MaxPool2d(kernel_size=2)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        outs = []
        for i in range(1, len(self._channels) + 1):
            # apply pooling after the first set of conv/att operations
            if i > 1:
                x = self.pooling(x)
            x = getattr(self, f"conv{i}")(x)
            x = getattr(self, f"att{i}")(x)
            outs.append(x)

        return outs


class Decoder(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, channels: list[int], kernel_size: int = 3, padding: int = 1
    ):
        super(Decoder, self).__init__()
        self._channels = channels
        channel_list = [in_channels] + channels

        for i, (in_ch, out_ch) in enumerate(itertools.pairwise(channel_list), start=1):
            setattr(self, f"upconv{i}", UpConv2d(in_ch, out_ch, kernel_size=2, stride=2))
            setattr(self, f"conv{i}", DoubleConv2d(in_ch, out_ch, kernel_size, padding=padding))
            setattr(self, f"ca{i}", ChannelAttention(out_ch))
            setattr(self, f"sa{i}", SpatialAttention())
            setattr(
                self, f"out_conv_{i}", nn.Conv2d(out_ch, out_channels, kernel_size=1, stride=1, padding=0, bias=True)
            )

    def forward(self, x: list[torch.Tensor]) -> list[torch.Tensor]:
        x, aux_inputs = x[-1], x[:-1]
        outs = []
        for i in range(1, len(self._channels) + 1):
            x = getattr(self, f"upconv{i}")(x)
            x = torch.cat([x, aux_inputs[-i]], dim=1)
            x = getattr(self, f"conv{i}")(x)
            x = getattr(self, f"ca{i}")(x) * x
            x = getattr(self, f"sa{i}")(x) * x
            outs.append(getattr(self, f"out_conv_{i}")(x))

        return outs[::-1]


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, dilation=1):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias, dilation=dilation
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class UpConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(UpConv2d, self).__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class DoubleConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(DoubleConv2d, self).__init__()
        self.conv1 = Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias)
        self.conv2 = Conv2d(out_channels, out_channels, kernel_size, stride, padding, bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class AttentionGroup(nn.Module):
    def __init__(self, num_channels):
        super(AttentionGroup, self).__init__()
        self.conv1 = Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.conv2 = Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.conv3 = Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.conv_1x1 = nn.Conv2d(num_channels, 3, kernel_size=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        s = torch.softmax(self.conv_1x1(x), dim=1)

        att = s[:, 0, :, :].unsqueeze(1) * x1 + s[:, 1, :, :].unsqueeze(1) * x2 + s[:, 2, :, :].unsqueeze(1) * x3

        return x + att


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // 16, in_planes, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class UnetAttention(nn.Module):
    def __init__(self):
        super(UnetAttention, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        out1, out2, out3, out4, x = self.encoder(x.float())
        x, aux_128, aux_64, aux_32 = self.decoder(out1, out2, out3, out4, x)

        return x.squeeze(), aux_128.squeeze(), aux_64.squeeze(), aux_32.squeeze()


if __name__ == "__main__":
    model = Encoder(in_channels=1, channels=[64, 128, 256, 512, 1024])
    x = torch.randn(1, 1, 512, 512)
    outs = model(x)
    print([y.shape for y in outs])
    model2 = Decoder(
        in_channels=outs[-1].shape[1],
        out_channels=1,
        channels=[512, 256, 128, 64],
    )
    outs2 = model2(outs)
    print([y.shape for y in outs2])

    model = UNetModule()
    x = torch.randn(1, 1, 512, 512)
    outs = model(x)
    print([y.shape for y in outs])
