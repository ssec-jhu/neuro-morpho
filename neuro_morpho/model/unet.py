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
from typing import Any

import cv2
import gin
import numpy as np
import torch
import torch.utils.data as td
from torch import nn
from tqdm import tqdm
from typing_extensions import override

import neuro_morpho.logging.base as base_logging
from neuro_morpho.data import data_loader
from neuro_morpho.model import base, loss, metrics
from neuro_morpho.util import TilesMixin, get_device

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
    return tensor[idx].detach().cpu().numpy()


@gin.register
class UNet(base.BaseModel, TilesMixin):
    def __init__(
        self,
        n_input_channels: int = 1,
        n_output_channels: int = 1,
        encoder_channels: list[int] = [64, 128, 256, 512, 1024],
        decoder_channels: list[int] = [512, 256, 128, 64],
        device: str = get_device(),
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
        self.exp_id: str = None

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

                x = self.cast_fn(x)  # b, 1, h, w
                # b, n_lbls, h, w
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

                    # select a random sample from the batch
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
        x = np.squeeze(x)
        image_size = x.shape
        image_tiles = self.tile_image(x)
        image_tiles = np.squeeze(image_tiles, axis=-1)

        n_y = len(self.y_coords)
        n_x = len(self.x_coords)
        pred_array = np.zeros((n_x * n_y, image_size[0], image_size[1]), dtype=np.float32)
        for i in range(n_y):
            for j in range(n_x):
                tile = image_tiles[i * n_x + j, :, :]
                # Start the inferring process
                tile_flip_0 = tile[::-1, ...]  # Vertical flip
                tile_flip_1 = tile[:, ::-1, ...]  # Horizontal flip
                tile_flip__1 = tile[::-1, ::-1, ...]  # Both axes
                tile_stack = np.stack([tile, tile_flip_0, tile_flip_1, tile_flip__1])
                tile_torch = torch.tensor(tile_stack).unsqueeze(1).to(torch.float32).to(self.device)
                with torch.no_grad():
                    pred, _, _, _ = self.model(tile_torch)
                    pred = torch.sigmoid(pred)
                    pred_ori, pred_flip_0, pred_flip_1, pred_flip__1 = pred
                pred_ori = pred_ori.cpu().numpy()
                pred_flip_0 = pred_flip_0.cpu().numpy()[::-1, ...]
                pred_flip_1 = pred_flip_1.cpu().numpy()[:, ::-1, ...]
                pred_flip__1 = pred_flip__1.cpu().numpy()[::-1, ::-1, ...]
                tile_pred = np.mean([pred_ori, pred_flip_0, pred_flip_1, pred_flip__1], axis=0)
                pred_array[
                    i * n_x + j,
                    self.y_coords[i] : (self.y_coords[i] + self.tile_size),
                    self.x_coords[j] : (self.x_coords[j] + self.tile_size),
                ] = tile_pred

        # Averaging the result
        non_zero_mask = pred_array != 0  # Shape (n_x * n_y, img_height, img_width)
        non_zero_count = np.sum(non_zero_mask, axis=0)  # Shape (img_height, img_width)
        non_zero_count[non_zero_count == 0] = 1  # Prevent division by zero
        if self.tile_assembly == "mean":
            non_zero_sum = np.sum(pred_array * non_zero_mask, axis=0)  # Shape (img_height, img_width)
            pred = non_zero_sum / non_zero_count  # Shape (img_height, img_width)
        elif self.tile_assembly == "max":
            pred = np.max(pred_array * non_zero_mask, axis=0)
        elif self.tile_assembly == "nn":  # nearest neighbor
            pred = np.zeros(image_size, dtype=np.float32)
            for idx in range(n_y * n_x):
                pred[self.nearest_map == idx] = pred_array[idx, self.nearest_map == idx]
        else:
            pred = np.zeros(image_size, dtype=np.float32)
            raise ValueError(f"Unknown tile assembly method: {self.tile_assembly}")

        return pred[np.newaxis, :, :]

    @override
    def predict_dir(self, in_dir: str | Path, out_dir: str | Path) -> None:
        in_dir = Path(in_dir)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        img_paths = list(Path(in_dir).glob("*.tif"))
        for img_path in img_paths:
            img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
            image = cv2.convertScaleAbs(img, alpha=255.0 / img.max()) / 255.0
            # Convert to shape (1, image.shape[0], image.shape[1], 1) => 1 sample, 1 channel
            x = image[np.newaxis, :, :, np.newaxis]
            pred = self.predict_proba(x)
            pred = (np.squeeze(pred) * 255).astype(np.uint8)
            pred_path = out_dir / f"{img_path.stem}_pred{img_path.suffix}"
            cv2.imwrite(pred_path, pred)

    @override
    def save(self, path: str | Path) -> None:
        save_path = Path(path) / (self.exp_id + ".pt" if self.exp_id else "model.pt")
        torch.save(self.model.state_dict(), save_path)

    @override
    def load(self, path: str | Path) -> None:
        self.model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))


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


@gin.configurable(allowlist=["ratio"])
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False),
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
