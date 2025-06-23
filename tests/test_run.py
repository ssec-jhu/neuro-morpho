from pathlib import Path

import cv2
import numpy as np

from neuro_morpho import run
from neuro_morpho.model.unet import UNet
from neuro_morpho.util import get_device


def test_run_infer():
    """Test the run function in the inference mode."""

    # Initialize the UNet model
    unet_model = UNet(
        n_input_channels=1,
        n_output_channels=1,
        encoder_channels=[64, 128, 256, 512, 1024],
        decoder_channels=[512, 256, 128, 64],
        device=get_device(),
    )

    testing_x_dir = "data/processed/test/"
    model_out_y_dir = "data/output/"

    # Create a dummy input image with the shape (batch_size, height, width, channels)
    input_tensor = np.random.rand(1, 1, 256, 256).astype(np.float32)
    Path(testing_x_dir).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(testing_x_dir + "test_img.tif", (input_tensor[0, 0, :, :] * 255).astype(np.uint8))

    run.run(
        model=unet_model,
        model_file=None,
        training_x_dir="data/processed/train/imgs/",
        training_y_dir="data/processed/train/lbls/",
        testing_x_dir=testing_x_dir,
        testing_y_dir="data/processed/test/lbls/",
        model_save_dir="models/",
        model_out_y_dir=model_out_y_dir,
        model_stats_output_dir="data/stats/model/",
        labeled_stats_output_dir="data/stats/label/",
        report_output_dir="data/report/",
        logger=None,
        train=False,
        infer=True,
        tile_size=512,
        tile_assembly="nn",
    )

    assert Path(model_out_y_dir).exists()
    assert (Path(model_out_y_dir) / Path("test_img_pred.tif")).exists()
