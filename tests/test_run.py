import itertools
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

    testing_dir = Path("data/processed/test")
    training_dir = Path("data/processed/train")
    model_out_y_dir = Path("data/output")

    for dir_path, subdir in itertools.product([testing_dir, training_dir], ["imgs", "lbls"]):
        (dir_path / subdir).mkdir(parents=True, exist_ok=True)

    input_tensor = np.random.rand(1, 1, 256, 256).astype(np.float32)
    cv2.imwrite(testing_dir / "imgs" / "test_img.tif", (input_tensor[0, 0, :, :] * 255).astype(np.uint8))

    run.run(
        model=unet_model,
        model_file=None,
        training_x_dir=training_dir / "imgs",
        training_y_dir=training_dir / "lbls",
        testing_x_dir=testing_dir / "imgs",
        testing_y_dir=testing_dir / "lbls",
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
