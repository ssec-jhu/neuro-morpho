import itertools
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest

from neuro_morpho import run
from neuro_morpho.logging.comet import CometLogger
from neuro_morpho.model.unet import UNet
from neuro_morpho.util import get_device


def test_run():
    """Test the run function in both test and inference modes."""

    # Initialize the UNet model
    unet_model = UNet(
        n_input_channels=1,
        n_output_channels=1,
        encoder_channels=[64, 128, 256, 512, 1024],
        decoder_channels=[512, 256, 128, 64],
        device=get_device(),
    )
    model_save_dir = Path("models")
    model_id = "019cae52c3654ebe8deec73b171b1e00"
    model_dir = model_save_dir / Path(model_id)
    checkpoints_dir = model_dir / Path("checkpoints")
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    unet_model.save(checkpoints_dir / "checkpoint_1.pt")

    unet_model.save_threshold(model_dir, 0.5)

    logger = CometLogger(experiment_key=model_id, disabled=True)

    training_dir = Path("data/processed/train")
    validating_dir = Path("data/processed/val")
    testing_dir = Path("data/processed/test")
    model_out_y_dir = Path("data/output") / Path(model_id)
    for dir_path, subdir in itertools.product([training_dir, validating_dir, testing_dir], ["test_imgs", "test_lbls"]):
        (dir_path / subdir).mkdir(parents=True, exist_ok=True)

    input_tensor = np.zeros((256, 256), dtype=np.float32)
    np.fill_diagonal(input_tensor, 1.0)
    np.fill_diagonal(np.fliplr(input_tensor), 1.0)
    # Cut a 4x4 black window in the center
    center_x, center_y = input_tensor.shape[0] // 2, input_tensor.shape[1] // 2
    half_window = 2  # because 4x4 window -> 2 pixels in each direction from center
    input_tensor[center_y - half_window : center_y + half_window, center_x - half_window : center_x + half_window] = 0.0
    target_tensor = input_tensor.copy()
    input_tensor = cv2.GaussianBlur(input_tensor, ksize=(3, 3), sigmaX=1)  # Apply Gaussian blur
    cv2.imwrite(training_dir / "test_imgs" / "test_img.tif", (input_tensor * 255).astype(np.uint8))
    cv2.imwrite(training_dir / "test_lbls" / "test_lbl.tif", (target_tensor * 255).astype(np.uint8))
    cv2.imwrite(validating_dir / "test_imgs" / "test_img.tif", (input_tensor * 255).astype(np.uint8))
    cv2.imwrite(validating_dir / "test_lbls" / "test_lbl.tif", (target_tensor * 255).astype(np.uint8))
    cv2.imwrite(testing_dir / "test_imgs" / "test_img.tif", (input_tensor * 255).astype(np.uint8))
    cv2.imwrite(testing_dir / "test_lbls" / "test_lbl.tif", (target_tensor * 255).astype(np.uint8))

    run.run(
        model=unet_model,
        training_x_dir=training_dir / "test_imgs",
        training_y_dir=training_dir / "test_lbls",
        validating_x_dir=validating_dir / "test_imgs",
        validating_y_dir=validating_dir / "test_lbls",
        testing_x_dir=testing_dir / "test_imgs",
        testing_y_dir=testing_dir / "test_lbls",
        model_save_dir=model_save_dir,
        model_out_y_dir=model_out_y_dir,
        model_stats_output_dir="data/stats/model/",
        labeled_stats_output_dir="data/stats/label/",
        report_output_dir="data/report/",
        logger=logger,
        train=False,
        get_threshold=False,
        test=True,
        infer=False,
    )

    assert Path(model_out_y_dir).exists()
    assert (Path(model_out_y_dir) / Path("test_img_pred.tif")).exists()
    assert (Path(model_out_y_dir) / Path("test_img_pred_bin.tif")).exists()
    assert (Path(model_out_y_dir) / Path("test_img_pred_bin_fixed.tif")).exists()
    shutil.rmtree(model_out_y_dir)

    run.run(
        model=unet_model,
        training_x_dir=training_dir / "train_imgs",
        training_y_dir=training_dir / "train_lbls",
        validating_x_dir=validating_dir / "test_imgs",
        validating_y_dir=validating_dir / "test_lbls",
        testing_x_dir=testing_dir / "test_imgs",
        testing_y_dir=testing_dir / "test_lbls",
        model_save_dir=model_save_dir,
        model_out_y_dir=model_out_y_dir,
        model_stats_output_dir="data/stats/model/",
        labeled_stats_output_dir="data/stats/label/",
        report_output_dir="data/report/",
        logger=logger,
        train=False,
        get_threshold=True,
        test=False,
        infer=True,
    )

    assert Path(model_out_y_dir).exists()
    assert (Path(model_out_y_dir) / Path("test_img_pred.tif")).exists()
    assert (Path(model_out_y_dir) / Path("test_img_pred_bin.tif")).exists()
    assert (Path(model_out_y_dir) / Path("test_img_pred_bin_fixed.tif")).exists()
    shutil.rmtree(model_out_y_dir)

    with pytest.raises(ValueError, match="Logger is not provided. Please provide a logger to log the results."):
        run.run(
            model=unet_model,
            training_x_dir=training_dir / "train_imgs",
            training_y_dir=training_dir / "train_lbls",
            validating_x_dir=validating_dir / "test_imgs",
            validating_y_dir=validating_dir / "test_lbls",
            testing_x_dir=testing_dir / "test_imgs",
            testing_y_dir=testing_dir / "test_lbls",
            model_save_dir=model_save_dir,
            model_out_y_dir=model_out_y_dir,
            model_stats_output_dir="data/stats/model/",
            labeled_stats_output_dir="data/stats/label/",
            report_output_dir="data/report/",
            logger=None,
            train=False,
            get_threshold=True,
            test=False,
            infer=True,
        )
