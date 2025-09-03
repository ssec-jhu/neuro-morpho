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
"""U-Net model for image segmentation."""

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
from sklearn.metrics import f1_score, confusion_matrix
from torch import nn
from tqdm import tqdm
from typing_extensions import override

import neuro_morpho.logging.base as base_logging
from neuro_morpho.data import data_loader
from neuro_morpho.model import base, loss, metrics
from neuro_morpho.model.breaks_analyzer import BreaksAnalyzer
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

    pred = model(x)

    losses = loss_fn(pred, y)
    loss = sum(map(lambda lss: lss[1], losses)) if isinstance(losses[0], (tuple, list)) else losses[1]

    loss.backward()

    optimizer.step()

    return pred, losses


def val_step(
    model: torch.nn.Module,
    loss_fn: loss.LOSS_FN,
    x: torch.Tensor,
    y: torch.Tensor,
) -> tuple[torch.Tensor, list[tuple[str, torch.Tensor]]]:
    """Perform a single validating step."""
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


def maybe_pbar(iterable, desc: str, unit: str, position: int, steps_bar: bool) -> tqdm:
    """Return a tqdm progress bar if steps_bar is True, otherwise return the iterable."""
    if steps_bar:
        return tqdm(iterable, desc=desc, unit=unit, position=position)
    return iterable

def global_f1(preds, labels):
    total_tp = total_fp = total_fn = 0

    for p, l in zip(preds, labels):  # iterate image by image
        tn, fp, fn, tp = confusion_matrix(l.ravel(), p.ravel(), labels=[0,1]).ravel()
        total_tp += tp
        total_fp += fp
        total_fn += fn

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0
    recall    = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0
    return 2 * precision * recall / (precision + recall) if (precision + recall) else 0
@gin.register
class UNet(base.BaseModel):
    """U-Net model for image segmentation.

    This class implements the U-Net architecture, a popular convolutional neural
    network for biomedical image segmentation.
    """

    def __init__(
        self,
        n_input_channels: int = 1,
        n_output_channels: int = 1,
        encoder_channels: list[int] = [64, 128, 256, 512, 1024],
        decoder_channels: list[int] = [512, 256, 128, 64],
        device: str = get_device(),
    ):
        """Initialize the UNet model.

        The architecture and code are adapted from:
        https://github.com/namdvt/skeletonization
        https://openaccess.thecvf.com/content/ICCV2021W/DLGC/html/Nguyen_U-Net_Based_Skeletonization_and_Bag_of_Tricks_ICCVW_2021_paper.html

        Args:
            n_input_channels (int, optional): Number of input channels. Defaults to 1.
            n_output_channels (int, optional): Number of output channels. Defaults to 1.
            encoder_channels (list[int], optional): List of channel sizes for the encoder.
                Defaults to [64, 128, 256, 512, 1024].
            decoder_channels (list[int], optional): List of channel sizes for the decoder.
                Defaults to [512, 256, 128, 64].
            device (str, optional): The device to run the model on. Defaults to get_device().
        """
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
            "train_data_loader_fn",
            "validate_data_loader_fn",
            "epochs",
            "optimizer",
            "loss_fn",
            "metric_fns",
            "logger",
            "log_every",
            "init_step",
            "model_id",
            "models_dir",
            "steps_bar",
        ]
    )
    def fit(
        self,
        training_x_dir: str | Path,
        training_y_dir: str | Path,
        validating_x_dir: str | Path,
        validating_y_dir: str | Path,
        train_data_loader_fn: Callable[[tuple[Path, Path]], td.DataLoader] | None = None,
        validate_data_loader_fn: Callable[[tuple[Path, Path]], td.DataLoader] | None = None,
        epochs: int = 1,
        optimizer: torch.optim.Optimizer = None,
        loss_fn: loss.LOSS_FN = None,
        metric_fns: list[metrics.METRIC_FN] | None = None,
        logger: base_logging.Logger = None,
        log_every: int = 10,
        init_step: int = 0,
        model_id: str | None = None,
        models_dir: str | Path = Path("models"),
        n_checkpoints: int = 5,  # Number of checkpoints to keep
        steps_bar: bool = True,  # Show progress bar during training/validating
    ) -> base.BaseModel:
        """Train the U-Net model.

        Args:
            training_x_dir (str | Path): Path to the training input images.
            training_y_dir (str | Path): Path to the training label images.
            validating_x_dir (str | Path): Path to the validating input images.
            validating_y_dir (str | Path): Path to the validating label images.
            train_data_loader_fn (Callable[[tuple[Path, Path]], td.DataLoader], optional): A function that returns
                the dataloader for the training set.
            validate_data_loader_fn (Callable[[tuple[Path, Path]], td.DataLoader], optional): A function that returns
                the dataloader forthe validation set.
            epochs (int, optional): Number of epochs to train for. Defaults to 1.
            optimizer (torch.optim.Optimizer, optional): The optimizer to use. Defaults to None.
            loss_fn (loss.LOSS_FN, optional): The loss function to use. Defaults to None.
            metric_fns (list[metrics.METRIC_FN] | None, optional): List of metric functions to use. Defaults to None.
            logger (base_logging.Logger, optional): The logger to use. Defaults to None.
            log_every (int, optional): Log every `log_every` steps. Defaults to 10.
            init_step (int, optional): The initial step number. Defaults to 0.
            model_id (str | None, optional): The ID of the model. Defaults to None.
            models_dir (str | Path, optional): The directory to save the models in. Defaults to Path("models").
            n_checkpoints (int, optional): The number of checkpoints to keep. Defaults to 5.
            steps_bar (bool, optional): Whether to show a progress bar for steps. Defaults to True.

        Returns:
            base.BaseModel: The trained model.
        """
        model_id = model_id or str(uuid.uuid4()).replace("-", "")

        model_dir = Path(models_dir) / model_id
        checkpoint_dir = model_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.optimizer = optimizer(params=self.model.parameters())
        self.load_checkpoint(checkpoint_dir)
        step = self.step if hasattr(self, "step") else init_step

        train_dl_fn = train_data_loader_fn or data_loader.build_dataloader
        validate_dl_fn = validate_data_loader_fn or data_loader.build_dataloader

        train_data_loader = train_dl_fn(training_x_dir, training_y_dir)
        validate_data_loader = validate_dl_fn(validating_x_dir, validating_y_dir)

        for _ in tqdm(range(epochs), desc="Epochs", unit="epoch", position=0):
            self.model.train()
            # x: b, 1, h, w
            # y: b, n_lbls, h, w
            training_iter = itertools.starmap(lambda x, y: (self.cast_fn(x), self.cast_fn(y)), train_data_loader)
            for x, y in maybe_pbar(training_iter, desc="Training", unit="batch", position=1, steps_bar=steps_bar):
                pred, losses = train_step(
                    model=self.model,
                    optimizer=self.optimizer,
                    loss_fn=loss_fn,
                    x=x,
                    y=y,
                )

                if logger is not None and step % log_every == 0:
                    self.save_checkpoint(checkpoint_dir, n_checkpoints, step)

                    x = detach_and_move(x, idx=0 if isinstance(x, tuple | list) else None)
                    y = detach_and_move(y, idx=0 if isinstance(y, tuple | list) else None)
                    pred = detach_and_move(pred, idx=0 if isinstance(pred, tuple | list) else None)
                    pred = 1 / (1 + np.exp(-pred))  # Sigmoid activation

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
                val_iter = itertools.starmap(lambda x, y: (self.cast_fn(x), self.cast_fn(y)), validate_data_loader)
                for x, y in maybe_pbar(val_iter, desc="Testing", unit="batch", position=2, steps_bar=steps_bar):
                    pred, losses = val_step(
                        model=self.model,
                        loss_fn=loss_fn,
                        x=x,
                        y=y,
                    )
                    x = detach_and_move(x, idx=0 if isinstance(x, tuple | list) else None)
                    y = detach_and_move(y, idx=0 if isinstance(y, tuple | list) else None)
                    pred = detach_and_move(pred, idx=0 if isinstance(pred, tuple | list) else None)
                    pred = 1 / (1 + np.exp(-pred))  # Sigmoid activation

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
        """Predict the probability map for an input image.

        This method uses tiling to handle large images. The tiles are processed
        by the model and then stitched back together.

        Args:
            x (np.ndarray): The input image.
            tiler (Tiler): The tiler to use for tiling the image.

        Returns:
            np.ndarray: The predicted probability map.
        """
        x = np.squeeze(x, axis=(0, 1))  # Remove batch_size and channels from (batch, channels, height, width)
        image_size = x.shape  # (height, width)
        image_tiles = tiler.tile_image(x)

        n_x, n_y = len(tiler.x_coords), len(tiler.y_coords)
        pred_array = np.zeros((n_x * n_y, image_size[0], image_size[1]), dtype=np.float32)
        self.model.eval()
        for i in range(n_y):
            for j in range(n_x):
                tile = image_tiles[i * n_x + j, :, :]
                # Start the inferring process
                tile_flip_0 = cv2.flip(tile, 0)  # Vertical flip
                tile_flip_1 = cv2.flip(tile, 1)  # Horizontal flip
                tile_flip__1 = cv2.flip(tile, -1)  # Both axes
                tile_stack = np.stack([tile, tile_flip_0, tile_flip_1, tile_flip__1])
                tile_torch = torch.tensor(tile_stack).unsqueeze(1).to(torch.float32).to(self.device)
                with torch.no_grad():
                    pred, _, _, _ = self.model(tile_torch)
                    pred = torch.sigmoid(pred)
                    pred_ori, pred_flip_0, pred_flip_1, pred_flip__1 = pred
                pred_ori = pred_ori.cpu().numpy().squeeze()
                pred_flip_0 = cv2.flip(pred_flip_0.cpu().numpy().squeeze(), 0)
                pred_flip_1 = cv2.flip(pred_flip_1.cpu().numpy().squeeze(), 1)
                pred_flip__1 = cv2.flip(pred_flip__1.cpu().numpy().squeeze(), -1)
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

    @gin.register(
        allowlist=[
            "tile_size",
            "tile_assembly",
            "binarize",
            "fix_breaks",
        ]
    )
    def predict_dir(
        self,
        in_dir: str | Path,
        out_dir: str | Path,
        threshold: float,
        mode: str,
        tile_size: tuple[int, int] = (512, 512),
        tile_assembly: str = "nn",
        binarize: bool = True,
        fix_breaks: bool = True,
    ) -> None:
        """Predict segmentations for all images in a directory.

        This method will predict the probability map for each image, then
        optionally binarize the result and analyze the breaks in the
        segmentation.

        Args:
            in_dir (str | Path): The directory containing the input images.
            out_dir (str | Path): The directory to save the predictions in.
            threshold (float): Use to get the hard prediction (binary output)
            mode (str): The mode of the prediction, can be 'test' or 'infer'
                'test' - runs the model on the test set (same size images) and saves the statistics
                'infer' - runs the model on the inference set (images may be of different size) and saves the output
            tile_size (tuple[int, int]): The size of the tiles to use for tiling the input images
            tile_assembly (str): The method for assembling the tiles, can be 'nn' (nearest neighbor), 'mean', or 'max'
            binarize (bool): Whether to binarize the output
            fix_breaks (bool): Whether to fix breaks in the binarized output
        """
        in_dir = Path(in_dir)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        img_paths = sorted(list(Path(in_dir).glob("*.tif")) + list(Path(in_dir).glob("*.pgm")))
        if not img_paths:
            raise ValueError(f"No images found in the input directory {in_dir}.")

        # Create tiler with the specified tile size and assembly method
        tiler = Tiler(tile_size[0], tile_assembly)
        if mode == "test":  # all images are of the same size
            image_size = cv2.imread(str(img_paths[0]), cv2.IMREAD_UNCHANGED).shape[:2]
            tiler.get_tiling_attributes(image_size)

        with tqdm(
            img_paths,
            total=len(img_paths),
            desc="Inferring images for prediction purposes",
            dynamic_ncols=True,
            leave=False,  # prevents lingering duplicate line
        ) as pbar:
            for img_path in pbar:
                pbar.set_postfix(file=img_path.name, refresh=False)
                image_shape_changed = False
                img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
                image = cv2.convertScaleAbs(img, alpha=255.0 / img.max()) / 255.0
                if mode == "infer":  # Extend image size if less then tile size and create tiling attributes
                    if image.shape[0] < tiler.tile_size or image.shape[1] < tiler.tile_size:  # Image is too small
                        image, crop_coord = tiler.extend_image_shape(image)  # Adjust image shape
                        image_shape_changed = True
                    tiler.get_tiling_attributes(image.shape[:2])
                # Get soft prediction for the image
                print("Getting soft prediction for the image ", img_path.name)
                image = np.stack(image)[np.newaxis, np.newaxis, :, :]
                pred = self.predict_proba(image, tiler)
                pred = np.squeeze(pred, axis=(0, 1))
                if image_shape_changed:
                    pred = pred[crop_coord[0] : crop_coord[0] + img.shape[0], crop_coord[1] : crop_coord[1] + img.shape[1]]
                pred_path = out_dir / f"{img_path.stem}_pred{img_path.suffix}"
                cv2.imwrite(pred_path, (pred * 255).astype(np.uint8))

                if binarize:  # Get hard prediction for the image
                    print("Getting hard prediction for the image ", pred_path.name)
                    pred_bin = pred.copy()
                    pred_bin[pred_bin >= threshold] = 1
                    pred_bin[pred_bin < threshold] = 0
                    pred_bin = (pred_bin * 255).astype(np.uint8)
                    pred_bin_path = out_dir / f"{img_path.stem}_pred_bin{img_path.suffix}"
                    cv2.imwrite(pred_bin_path, pred_bin)

                if fix_breaks:
                    breaks_analyzer = BreaksAnalyzer()
                    print("Fixing breaks for the image ", pred_bin_path.name)
                    pred_bin_fixed_img = breaks_analyzer.analyze_breaks(pred_bin, pred).copy()
                    pred_bin_fixed_path = out_dir / f"{img_path.stem}_pred_bin_fixed{img_path.suffix}"
                    cv2.imwrite(pred_bin_fixed_path, pred_bin_fixed_img)

    @gin.register(
        allowlist=[
            "tile_size",
            "tile_assembly",
        ]
    )
    def find_threshold(
        self,
        img_dir: str | Path,
        lbl_dir: str | Path,
        model_dir: str | Path,
        model_out_val_y_dir: str | Path,
        tile_size: tuple[int, int] = (512, 512),
        tile_assembly: str = "nn",
    ) -> float:
        """Find the optimal threshold for binarizing a soft prediction.

        Args:
            img_dir (str | Path | None, optional): Directory containing the
                original images. Defaults to None.
            lbl_dir (str | Path | None, optional): Directory containing
                the ground truth segmentations. Defaults to None.
            tiler (Tiler, optional): Tiler object for tiling the images.
                Defaults to None.

        Returns:
            float: The optimal threshold.
        """
        threshold = self.load_threshold(model_dir)
        if threshold is not None:
            return threshold

        if img_dir is None or lbl_dir is None:
            raise ValueError("Both image and label directories must be provided.")

        img_dir = Path(img_dir)
        lbl_dir = Path(lbl_dir)
        model_out_val_y_dir = Path(model_out_val_y_dir)

        img_paths = sorted(list(Path(img_dir).glob("*.tif")) + list(Path(img_dir).glob("*.pgm")))
        lbl_paths = sorted(list(Path(lbl_dir).glob("*.tif")) + list(Path(lbl_dir).glob("*.pgm")))
        if not img_paths or not lbl_paths:
            raise ValueError("No images found in one or both of the provided directories.")

        # Ensure the number of images in both directories match
        if len(img_paths) != len(lbl_paths):
            raise ValueError("The number of images in the input and target directories must match.")

        compute_predictions = True # Whether to compute predictions or use existing ones
        if model_out_val_y_dir.exists():
            pred_paths = sorted(list(Path(model_out_val_y_dir).glob("*.tif")) + list(Path(model_out_val_y_dir).glob("*.pgm")))
            if len(pred_paths) != len(lbl_paths):
                warnings.warn(
                    f"Output validation directory {model_out_val_y_dir} already exists but has a different number of files "
                    f"({len(pred_paths)}) than the label validation directory {lbl_dir} ({len(lbl_paths)}). "
                    "Recomputing the predictions and overwriting the existing files."
                )
            else:
                print(f"Using existing predictions in {model_out_val_y_dir} to compute the optimal threshold.")
                compute_predictions = False
        else:
            model_out_val_y_dir.mkdir(parents=True, exist_ok=True)
        
        preds = list()
        if compute_predictions:
            # Create a tiler object to handle the tiling of the images
            tiler = Tiler(tile_size[0], tile_assembly)
            image = cv2.imread(str(img_paths[0]), cv2.IMREAD_UNCHANGED)
            tiler.get_tiling_attributes(image.shape[:2])  # Get tiling attributes based on image size

            # Read images and get predictions
            with tqdm(
                img_paths,
                total=len(img_paths),
                desc="Inferring images for threshold calculation",
                dynamic_ncols=True,
                leave=False,  # prevents lingering duplicate line
            ) as pbar:
                for img_path in pbar:
                    pbar.set_postfix(file=img_path.name, refresh=False)
                    if not img_path.exists():
                        raise FileNotFoundError(f"Image {img_path} does not exist.")
                    # Read the image and target
                    image = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
                    image = cv2.convertScaleAbs(image, alpha=255.0 / image.max()) / 255.0
                    if image is None:
                        raise ValueError(
                            f"Could not read image {img_path}. Ensure it is valid image file."
                        )
                    # Get soft prediction for the image
                    image = np.stack(image)[np.newaxis, np.newaxis, :, :]
                    pred = self.predict_proba(image, tiler)
                    if pred is None:
                        raise ValueError(f"Could not get soft prediction for image {img_path}.")
                    pred = np.squeeze(pred, axis=(0, 1))   
                    pred_path = model_out_val_y_dir / f"{img_path.stem}_pred{img_path.suffix}"
                    cv2.imwrite(pred_path, (pred * 255).astype(np.uint8))
                    preds.append(pred)
                    
        else:  # Load existing predictions
            with tqdm(
                pred_paths,
                total=len(pred_paths),
                desc="Loading predictions for threshold calculation",
                dynamic_ncols=True,
                leave=False,  # prevents lingering duplicate line
            ) as pbar:
                for pred_path in pbar:
                    pbar.set_postfix(file=pred_path.name, refresh=False)
                    pred = cv2.imread(str(pred_path), cv2.IMREAD_UNCHANGED)
                    if pred is None:
                        raise ValueError(
                            f"Could not read prediction {pred_path}. Ensure it is valid image file."
                        )
                    pred = (pred.astype(np.float32) / pred.max()).astype(np.float32)
                    preds.append(pred)
                
        preds = np.stack(preds)
        
        # Read labels        
        labels = list()
        with tqdm(
            lbl_paths,
            total=len(lbl_paths),
            desc="Loading labels for threshold calculation",
            dynamic_ncols=True,
            leave=False,  # prevents lingering duplicate line
        ) as pbar:
            for lbl_path in pbar:
                pbar.set_postfix(file=lbl_path.name, refresh=False)
                label = cv2.imread(str(lbl_path), cv2.IMREAD_UNCHANGED)
                if label is None:
                    raise ValueError(
                        f"Could not read label {lbl_path}. Ensure it is valid image file."
                    )
                label = (label.astype(np.float32) / label.max()).astype(np.uint8)
                labels.append(label)
        labels = np.stack(labels)

        # Calculate the optimal threshold
        f1s = list()
        thresholds = np.stack(list(range(20, 70))) / 100
        with tqdm(
            thresholds,
            total=len(thresholds),
            dynamic_ncols=True,
            leave=False,  # prevents lingering duplicate line
        ) as pbar:
            for threshold in pbar:
                # preds_ = preds.copy()
                # preds_[preds_ >= threshold] = 1
                # preds_[preds_ < threshold] = 0
                preds_bin = (preds >= threshold).astype(np.uint8)
                # f1s.append(f1_score(preds_bin.reshape(-1), labels.reshape(-1)))
                f1 = global_f1(preds_bin, labels)
                f1s.append(f1)
                pbar.set_description(f"Calculated f1 score for threshold {threshold:.2f} is {f1:.4f}", refresh=False)
    
        f1s = np.stack(f1s)
        threshold = thresholds[f1s.argmax()]
        self.save_threshold(model_dir, threshold)

        return threshold

    def save_checkpoint(self, checkpoint_dir: Path | str, n_checkpoints: int, step: int) -> None:
        """Save a checkpoint of the model.

        This method will save the model's state dict and the current step number.
        It will also remove old checkpoints to keep only the `n_checkpoints` most
        recent ones.

        Args:
            checkpoint_dir (Path | str): The directory to save the checkpoint in.
            n_checkpoints (int): The number of checkpoints to keep.
            step (int): The current step number.
        """
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoints = list(checkpoint_dir.glob("*.pt"))

        if len(checkpoints) >= n_checkpoints:
            need_to_remove = (len(checkpoints) - n_checkpoints) + 1
            checkpoints_to_remove = sorted(
                checkpoints,
                # st_mtime is the time of last modification: https://docs.python.org/3/library/stat.html#stat.ST_MTIME
                # we want to remove the oldest checkpoints so we sort by that.
                key=lambda p: p.stat().st_mtime,
            )[:need_to_remove]

            for ctr in checkpoints_to_remove:
                ctr.unlink()

        checkpoint_path = checkpoint_dir / f"checkpoint_{step}.pt"

        self.step = step
        self.save(checkpoint_path)

    def load_checkpoint(self, checkpoint_dir: Path | str) -> None:
        """Load the most recent checkpoint from a directory.

        Args:
            checkpoint_dir (Path | str): The directory containing the checkpoints.
        """
        checkpoint_dir = Path(checkpoint_dir)
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint dir {checkpoint_dir!s} does not exist")

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
        print(f"Loading checkpoint: {checkpoint!s}")
        self.load(checkpoint)

    @override
    def save(self, path: str | Path) -> None:
        """Save the model to a file.

        Args:
            path (str | Path): The path to save the model to.
        """
        path = Path(path)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "step": getattr(self, "step", 0),
                "optimizer_state_dict": self.optimizer.state_dict() if hasattr(self, "optimizer") else None,
            },
            path,
        )

    @override
    def load(self, path: str | Path) -> None:
        """Load the model from a file.

        Args:
            path (str | Path): The path to load the model from.
        """
        path = Path(path)
        # The model can be opened on CPU or Mac, so we use map_location to ensure that.
        data = torch.load(path, map_location=torch.device("cpu"))  # nosec B614: File is locally generated
        # and verified to contain only model state_dict.
        self.model.load_state_dict(data["model_state_dict"])
        self.model.to(self.device)
        self.step = data.get("step", 0)
        # if we're training we have an optimizer.
        # If not, we don't need to load the optimizer state_dict.
        if hasattr(self, "optimizer"):
            self.optimizer.load_state_dict(data["optimizer_state_dict"])

    def save_threshold(self, model_dir: Path | str, threshold: float) -> None:
        """Save a binarization threshold for a given model.

        This method will save the threshold to a file named `threshold.csv` in the
        specified model directory.

        Args:
            model_dir (Path | str): The directory to save the threshold file in.
            threshold (float): The threshold value to save.
        """
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        thresh_file_path = model_dir / "threshold.csv"
        with open(thresh_file_path, "w") as f:
            f.write(f"{threshold}\n")
            print(f"The threshold for the given model is found to be {threshold:.2f} and has been saved.")

    def load_threshold(self, model_dir: Path | str) -> float:
        """Load the threshold from a given model path.

        Args:
            model_dir (Path | str): The directory containing the threshold file.
        """
        model_dir = Path(model_dir)
        if not model_dir.exists():
            raise FileNotFoundError(f"Model dir {model_dir!s} does not exist")

        thresh_file_path = model_dir / "threshold.csv"
        if not thresh_file_path.exists():
            warnings.warn(f"Threshold file {thresh_file_path!s} does not exist")
            threshold = None
        else:
            with open(thresh_file_path) as f:
                threshold = float(f.read().strip())
                print(f"The threshold for the given model exists: {threshold:.2f} and has been loaded.")

        return threshold


class UNetModule(nn.Module):
    """The U-Net module.

    This module contains the encoder and decoder parts of the U-Net.
    """

    def __init__(
        self,
        n_input_channels: int = 1,
        n_output_channels: int = 1,
        encoder_channels: list[int] = [64, 128, 256, 512, 1024],
        decoder_channels: list[int] = [512, 256, 128, 64],
    ):
        """Initialize the UNetModule.

        Args:
            n_input_channels (int, optional): Number of input channels.
                Defaults to 1.
            n_output_channels (int, optional): Number of output channels.
                Defaults to 1.
            encoder_channels (list[int], optional): List of channel sizes for
                the encoder. Defaults to [64, 128, 256, 512, 1024].
            decoder_channels (list[int], optional): List of channel sizes for
                the decoder. Defaults to [512, 256, 128, 64].
        """
        super(UNetModule, self).__init__()
        self.encoder = Encoder(in_channels=n_input_channels, channels=encoder_channels)
        self.decoder = Decoder(
            in_channels=encoder_channels[-1],
            channels=decoder_channels,
            out_channels=n_output_channels,
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Forward pass through the U-Net.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            list[torch.Tensor]: A list of output tensors from the decoder.
        """
        x = self.encoder(x)
        return self.decoder(x)


class Encoder(nn.Module):
    """The encoder part of the U-Net.

    This module consists of a series of convolutional and attention layers
    followed by max pooling.
    """

    def __init__(self, in_channels: int, channels: list[int], kernel_size: int = 3, padding: int = 1):
        """Initialize the Encoder.

        Args:
            in_channels (int): The number of input channels.
            channels (list[int]): A list of the number of channels for each
                convolutional layer.
            kernel_size (int, optional): The size of the convolutional kernel.
                Defaults to 3.
            padding (int, optional): The padding for the convolution. Defaults
                to 1.
        """
        super(Encoder, self).__init__()
        self._channels = channels
        channel_list = [in_channels] + channels

        for i, (in_ch, out_ch) in enumerate(itertools.pairwise(channel_list), start=1):
            setattr(self, f"conv{i}", DoubleConv2d(in_ch, out_ch, kernel_size, padding=padding))
            setattr(self, f"att{i}", AttentionGroup(out_ch))
        self.pooling = nn.MaxPool2d(kernel_size=2)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Forward pass through the encoder.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            list[torch.Tensor]: A list of the output tensors from each block
                before pooling.
        """
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
    """The decoder part of the U-Net.

    This module consists of a series of up-convolutional, convolutional, and
    attention layers.
    """

    def __init__(
        self, in_channels: int, out_channels: int, channels: list[int], kernel_size: int = 3, padding: int = 1
    ):
        """Initialize the Decoder.

        Args:
            in_channels (int): The number of input channels.
            out_channels (int): The number of output channels.
            channels (list[int]): A list of the number of channels for each
                convolutional layer.
            kernel_size (int, optional): The size of the convolutional kernel.
                Defaults to 3.
            padding (int, optional): The padding for the convolution.
                Defaults to 1.
        """
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
        """Forward pass through the decoder.

        Args:
            x (list[torch.Tensor]): A list of the output tensors from the encoder.

        Returns:
            list[torch.Tensor]: A list of the output tensors from each block.
        """
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
    """A convolutional layer with batch normalization and ReLU activation."""

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
    """An up-convolutional layer with batch normalization and ReLU activation."""

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
    """A block of two convolutional layers."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(DoubleConv2d, self).__init__()
        self.conv1 = Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias)
        self.conv2 = Conv2d(out_channels, out_channels, kernel_size, stride, padding, bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class AttentionGroup(nn.Module):
    """An attention group module."""

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
    """A channel attention module."""

    def __init__(self, in_planes: int, ratio: int = 16):
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
    """A spatial attention module."""

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
