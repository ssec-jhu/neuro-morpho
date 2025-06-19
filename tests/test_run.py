from pathlib import Path

import cv2
import numpy as np

from neuro_morpho import run
from neuro_morpho.model.unet import UNet
from neuro_morpho.util import get_device


def test_run_infer():
    """Test the run function in the inference mode."""

    # Initialize config params
    run.train = False
    run.infer = True
    run.model = UNet(
        n_input_channels=1,
        n_output_channels=1,
        encoder_channels=[64, 128, 256, 512, 1024],
        decoder_channels=[512, 256, 128, 64],
        device=get_device(),
    )
    run.model_file = None
    run.training_x_dir = "data/processed/train/imgs/"
    run.training_y_dir = "data/processed/train/lbls/"
    run.testing_x_dir = "data/processed/test/"
    run.testing_y_dir = "data/processed/test/lbls/"
    run.model_save_dir = "models/"
    run.model_out_y_dir = "data/output/"
    run.model_stats_output_dir = "data/stats/model/"
    run.labeled_stats_output_dir = "data/stats/label/"
    run.report_output_dir = "data/report/"
    run.tile_size = 512
    run.tile_assembly = "nn"
    run.logger = None

    # Create a dummy input image with the shape (batch_size, height, width, channels)
    input_tensor = np.random.rand(1, 1, 256, 256).astype(np.float32)
    Path(run.testing_x_dir).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(run.testing_x_dir + "test_img.tif", (input_tensor[0, 0, :, :] * 255).astype(np.uint8))

    run.run(
        model=run.model,
        model_file=run.model_file,
        training_x_dir=run.training_x_dir,
        training_y_dir=run.training_y_dir,
        testing_x_dir=run.testing_x_dir,
        testing_y_dir=run.testing_y_dir,
        model_save_dir=run.model_save_dir,
        model_out_y_dir=run.model_out_y_dir,
        model_stats_output_dir=run.model_stats_output_dir,
        labeled_stats_output_dir=run.labeled_stats_output_dir,
        report_output_dir=run.report_output_dir,
        logger=run.logger,
        train=run.train,
        infer=run.infer,
        tile_size=run.tile_size,
        tile_assembly=run.tile_assembly,
    )

    assert Path(run.model_out_y_dir).exists()
    assert (Path(run.model_out_y_dir) / Path("test_img_pred.tif")).exists()
