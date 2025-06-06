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
import uuid
import warnings
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
from neuro_morpho.model.breaks_analyzer import BreaksAnalyzer
from neuro_morpho.model.threshold import ThresholdFinder
from neuro_morpho.model.tiler import Tiler
from neuro_morpho.util import get_device

ERR_PREDICT_DIR_NOT_IMPLEMENTED = (
    "The predict_dir method is not implemented, because you might be tiling, subclass and implement this method."
)


def apply_tpl(fn: Callable, item: Any | tuple[Any, ...]) -> Any | tuple:
    """Apply a function to a an item or to all of the items in a tuple."""
    return tuple(map(fn, item)) if isinstance(item, tuple | list) else fn(item)


def cast_and_move(tensor: torch.Tensor, device: str) -> torch.Tensor:
    """Cast and move tensor to the specified device."""
    return tensor.float().to(device)


def detach_and_move(tensor: torch.Tensor, idx: int | None = None) -> np.ndarray:
    """Detach and move tensor to the specified device."""
    if idx is None:
        return tensor.detach().cpu().numpy()
    return tensor[idx].detach().cpu().numpy()


def train_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: loss.LOSS_FN,
    x: torch.Tensor,
    y: torch.Tensor,
) -> tuple[torch.Tensor, list[tuple[str, torch.Tensor]]]:
    """Perform a single training step."""
    optimizer.zero_grad()

    # start = time()
    pred = model(x)
    # print("pred takes", time()-start)

    # start = time()
    losses = loss_fn(pred, y)
    # print("losses takes", time()-start)

    # start = time()
    loss = sum(map(lambda lss: lss[1], losses)) if isinstance(losses[0], (tuple, list)) else losses[1]
    # print("total loss takes", time()-start)

    # start = time()
    loss.backward()
    # print("loss takes", time()-start)

    # start = time()
    optimizer.step()
    # print("opt step takes", time()-start)

    return pred, losses


def test_step(
    model: torch.nn.Module,
    loss_fn: loss.LOSS_FN,
    x: torch.Tensor,
    y: torch.Tensor,
) -> tuple[torch.Tensor, list[tuple[str, torch.Tensor]]]:
    """Perform a single testing step."""
    with torch.no_grad():
        pred = model(x)
        losses = loss_fn(pred, y)

    return pred, losses


def log_metrics(
    logger: base_logging.Logger,
    metric_fns: list[metrics.METRIC_FN],
    pred: torch.Tensor,
    y: torch.Tensor,
    is_train: bool,
    step: int,
) -> None:
    """Log metrics to the logger."""
    metrics_values = [fn(pred, y) for fn in metric_fns]
    for name, value in metrics_values:
        logger.log_scalar(name, value, step=step, train=is_train)


def log_losses(
    logger: base_logging.Logger,
    losses: list[tuple[str, torch.Tensor]],
    total_loss: torch.Tensor,
    is_train: bool,
    step: int,
) -> None:
    """Log losses to the logger."""
    for name, value in losses:
        logger.log_scalar(name, value.item(), step=step, train=is_train)
    logger.log_scalar("loss", total_loss.item(), step=step, train=is_train)


def log_sample(
    logger: base_logging.Logger,
    x: torch.Tensor,
    y: torch.Tensor,
    pred: torch.Tensor,
    is_train: bool,
    step: int,
    idx: int | None = None,
) -> None:
    """Log a sample triplet (input, target, prediction) to the logger."""
    # select a random sample from the batch
    sample_idx = idx if idx is not None else np.random.choice(x.shape[0], size=1)[0]
    sample_x = x[sample_idx, ...].squeeze()
    sample_y = y[sample_idx, ...].squeeze()
    sample_pred = pred[sample_idx, ...].squeeze()

    logger.log_triplet(sample_x, sample_y, sample_pred, "triplet", step=step, train=is_train)


@gin.register
class UNet(base.BaseModel):
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
        self.cast_fn = functools.partial(apply_tpl, functools.partial(cast_and_move, device=device))
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
            "init_step",
            "model_id",
            "model_dir",
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
        model_id: str | None = None,
        model_dir: str | Path | None = None,
    ) -> base.BaseModel:
        if model_id and model_dir:
            model_path = Path(model_dir) / model_id
            if model_path.exists():
                self.load(model_path)
                print(f"Resumed training from model: {model_path}")
            else:
                raise FileNotFoundError(f"Model directory not found: {model_path}")

        step = self.step if hasattr(self, "step") else init_step

        if train_data_loader is None:
            train_data_loader = data_loader.build_dataloader(training_x_dir, training_y_dir)
        if test_data_loader is None:
            test_data_loader = data_loader.build_dataloader(testing_x_dir, testing_y_dir)

        optimizer = optimizer(params=self.model.parameters())

        # TODO: steps needs to be fixed
        for n_epoch in tqdm(range(epochs), desc="Epochs", unit="epoch", position=0):
            self.model.train()
            # x: b, 1, h, w
            # y: b, n_lbls, h, w
            for x, y in itertools.starmap(lambda x, y: (self.cast_fn(x), self.cast_fn(y)), train_data_loader):
                pred, losses = train_step(
                    model=self.model,
                    optimizer=optimizer,
                    loss_fn=loss_fn,
                    x=x,
                    y=y,
                )

                if logger is not None and step % log_every == 0:
                    self.save_checkpoint(checkpoint_dir, n_checkpoints, step)

                    x = detach_and_move(x, idx=0 if isinstance(x, tuple | list) else None)
                    y = detach_and_move(y, idx=0 if isinstance(y, tuple | list) else None)
                    pred = detach_and_move(pred, idx=0 if isinstance(pred, tuple | list) else None)

                    log_metrics(
                        logger=logger,
                        metric_fns=metric_fns,
                        pred=pred,
                        y=y,
                        is_train=True,
                        step=step,
                    )

                    log_losses(
                        logger=logger,
                        losses=losses,
                        total_loss=sum(map(lambda lss: lss[1], losses)),
                        is_train=True,
                        step=step,
                    )

                    log_sample(
                        logger=logger,
                        x=x,
                        y=y,
                        pred=pred,
                        is_train=True,
                        step=step,
                    )

                step += 1

            if logger is not None:
                self.save_checkpoint(checkpoint_dir, n_checkpoints, step)
                self.model.eval()
                scalars_numerator = defaultdict(float)
                scalars_denominator = defaultdict(float)

                # x: b, 1, h, w
                # y: b, n_lbls, h, w
                for x, y in itertools.starmap(lambda x, y: (self.cast_fn(x), self.cast_fn(y)), test_data_loader):
                    pred, losses = test_step(
                        model=self.model,
                        loss_fn=loss_fn,
                        x=x,
                        y=y,
                    )
                    x = detach_and_move(x, idx=0 if isinstance(x, tuple | list) else None)
                    pred = detach_and_move(pred, idx=0 if isinstance(pred, tuple | list) else None)
                    y = detach_and_move(y, idx=0 if isinstance(y, tuple | list) else None)

                    loss = sum(map(lambda lss: lss[1], losses)) if isinstance(losses, (tuple, list)) else losses[1]
                    metrics_values = [fn(pred, y) for fn in metric_fns]
                    for name, loss in losses + metrics_values + [("loss", loss)]:
                        scalars_numerator[name] += loss.item() * x.shape[0]
                        scalars_denominator[name] += x.shape[0]

                for name, num in scalars_numerator.items():
                    logger.log_scalar(name, num / scalars_denominator[name], step=step, train=False)

                log_sample(
                    logger=logger,
                    x=x,
                    y=y,
                    pred=pred,
                    is_train=False,
                    step=step,
                )

        # After all epochs save a copy in the models_dir
        self.save(model_dir / "model.pt")

        return self

    @override
    def predict_proba(self, x: np.ndarray, tiler: Tiler) -> np.ndarray:
        x = np.squeeze(x, axis=(0, 1))  # Remove batch_size and channels from (batch, channels, height, width)
        image_size = x.shape  # (height, width)
        image_tiles = tiler.tile_image(x)

        n_x, n_y = len(tiler.x_coords), len(tiler.y_coords)
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
                    tiler.y_coords[i] : (tiler.y_coords[i] + tiler.tile_size),
                    tiler.x_coords[j] : (tiler.x_coords[j] + tiler.tile_size),
                ] = tile_pred

        # Averaging the result
        non_zero_mask = pred_array != 0  # Shape (n_x * n_y, img_height, img_width)
        non_zero_count = np.sum(non_zero_mask, axis=0)  # Shape (img_height, img_width)
        non_zero_count[non_zero_count == 0] = 1  # Prevent division by zero
        if tiler.tile_assembly == "mean":
            non_zero_sum = np.sum(pred_array, axis=0)  # Shape (img_height, img_width)
            non_zero_sum = np.sum(pred_array * non_zero_mask, axis=0)  # Shape (img_height, img_width)
            pred = non_zero_sum / non_zero_count  # Shape (img_height, img_width)
        elif tiler.tile_assembly == "max":
            pred = np.max(pred_array * non_zero_mask, axis=0)
        elif tiler.tile_assembly == "nn":  # nearest neighbor
            pred = np.zeros(image_size, dtype=np.float32)
            for idx in range(n_y * n_x):
                pred[tiler.nearest_map == idx] = pred_array[idx, tiler.nearest_map == idx]
        else:
            pred = np.zeros(image_size, dtype=np.float32)
            raise ValueError(f"Unknown tile assembly method: {self.tile_assembly}")

        return pred[np.newaxis, np.newaxis, :, :]  # (1, 1, height, width)

    @override
    def predict_dir(
        self,
        in_dir: str | Path,
        out_dir: str | Path,
        tar_dir: str | Path,
        tiler: Tiler = None,
        binarize: bool = True,
        analyze: bool = True,
    ) -> None:
        in_dir = Path(in_dir)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        img_paths = sorted(list(Path(in_dir).glob("*.tif")) + list(Path(in_dir).glob("*.pgm")))
        for img_path in img_paths:
            img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
            image = cv2.convertScaleAbs(img, alpha=255.0 / img.max()) / 255.0
            # Convert to shape (1, 1, image.shape[0], image.shape[1]) => 1 sample, 1 channel
            image = np.stack(image)[np.newaxis, np.newaxis, :, :]
            pred = self.predict_proba(image, tiler)
            pred = (np.squeeze(pred, axis=(0, 1)) * 255).astype(np.uint8)
            pred_path = out_dir / f"{img_path.stem}_pred{img_path.suffix}"
            cv2.imwrite(pred_path, pred)

        if binarize:
            thresh = ThresholdFinder().find_threshold(out_dir, tar_dir, tiler)
            pred_paths = sorted(list(Path(out_dir).glob("*_pred.tif")) + list(Path(out_dir).glob("*_pred.pgm")))
            for pred_path in pred_paths:
                pred = cv2.imread(str(pred_path), cv2.IMREAD_UNCHANGED) / 255.0
                pred_bin = pred.copy()
                pred_bin[pred_bin >= thresh] = 1
                pred_bin[pred_bin < thresh] = 0
                pred_bin = (pred_bin * 255).astype(np.uint8)
                pred_bin_path = out_dir / f"{img_path.stem}_pred_bin{img_path.suffix}"
                cv2.imwrite(pred_bin_path, pred_bin)

            if analyze:
                breaks_analyzer = BreaksAnalyzer()
                pred_bin_paths = sorted(
                    list(Path(out_dir).glob("*_pred_bin.tif")) + list(Path(out_dir).glob("*_pred_bin.pgm"))
                )
                if not pred_bin_paths:
                    raise ValueError("No predicted binary images found for analysis.")
                pred_paths = sorted(list(Path(out_dir).glob("*_pred.tif")) + list(Path(out_dir).glob("*_pred.pgm")))
                if not pred_paths:
                    raise ValueError("No predicted images found for analysis.")
                if len(pred_bin_paths) != len(pred_paths):
                    raise ValueError(
                        "The number of predicted binary images does not match the number of predicted images. "
                        "Analysis will be skipped."
                    )
                for pred_bin_path, pred_path in zip(pred_bin_paths, pred_paths, strict=False):
                    pred_bin_img = cv2.imread(str(pred_bin_path), cv2.IMREAD_UNCHANGED)
                    pred_img = cv2.imread(str(pred_path), cv2.IMREAD_UNCHANGED)
                    pred_bin_fixed_img = breaks_analyzer.analyze_breaks(pred_bin_img, pred_img)
                    # Save the fixed image if needed
                    pred_bin_fixed_path = out_dir / f"{img_path.stem}_pred_bin_fixed{img_path.suffix}"
                    cv2.imwrite(pred_bin_fixed_path, pred_bin_fixed_img * 255)

    def save_checkpoint(self, checkpoint_dir: Path | str, n_checkpoints: int, step: int) -> None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoints = list(checkpoint_dir.glob("*.pt"))

        if len(checkpoints) >= n_checkpoints:
            need_to_remove = (len(checkpoints) - n_checkpoints) + 1
            checkpoints_to_remove = list(
                sorted(
                    checkpoints,
                    # st_mtime is the time of last modification: https://docs.python.org/3/library/stat.html#stat.ST_MTIME
                    # we want to remove the oldest checkpoints so we sort by that.
                    key=lambda p: p.stat().st_mtime,
                )
            )[:need_to_remove]

            for ctr in checkpoints_to_remove:
                ctr.unlink()

        checkpoint_path = checkpoint_dir / f"checkpoint_{step}.pt"

        self.step = step
        self.save(checkpoint_path)

    def load_checkpoint(self, checkpoint_dir: Path | str) -> None:
        checkpoint_dir = Path(checkpoint_dir)
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint dir {str(checkpoint_dir)} does not exist")

        checkpoints = list(checkpoint_dir.glob("checkpoint_*.pt"))

        if len(checkpoints) == 0:
            warnings.warn("No checkpoints found to load")
            return

        checkpoint = sorted(
            checkpoints,
            # st_mtime is the time of last modification: https://docs.python.org/3/library/stat.html#stat.ST_MTIME
            # we want to retrieve the latest checkpoint, so we reverse the sort
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )[0]
        print(f"Loading checkpoint: {str(checkpoint)}")
        self.load(checkpoint)

    @override
    def save(self, path: str | Path) -> None:
        save_dir = Path(path) / (self.exp_id if self.exp_id else "model")
        save_dir.mkdir(parents=True, exist_ok=True)

        model_path = save_dir / "model.pt"
        step_path = save_dir / "step.txt"

        torch.save(self.model.state_dict(), model_path)
        with open(step_path, "w") as f:
            f.write(str(self.step))

    @override
    def load(self, path: str | Path) -> None:
        load_dir = Path(path)
        model_path = load_dir / "model.pt"
        step_path = load_dir / "step.txt"

        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        if step_path.exists():
            with open(step_path, "r") as f:
                self.step = int(f.read())
        else:
            self.step = 0


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
