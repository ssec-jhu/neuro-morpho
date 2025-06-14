import numpy as np

from neuro_morpho import run
from neuro_morpho.model import unet
from neuro_morpho.util import get_device


def test_run_test():
    """Test the run function in test mode."""

    # Create a dummy input image with the shape (batch_size, height, width, channels)
    input_tensor = np.random.rand(1, 256, 256, 1).astype(np.float32)

    # Initialize config params
    run.train = False
    run.test = True
    run.training_x_dir = "dummy_training_x_dir"
    run.training_y_dir = "dummy_training_y_dir"
    run.testing_x_dir = "dummy_testing_x_dir"
    run.testing_y_dir = "dummy_testing_y_dir"
    run.model_save_dir = "dummy_model_save_dir"
    run.model_out_y_dir = "dummy_model_out_y_dir"
    run.model_stats_output_dir = "dummy_model_stats_output_dir"
    run.labled_stats_outpur_dir = "dummy_labled_stats_outpur_dir"
    run.report_output_dir = "dummy_report_output_dir"
    run.tile_size = 128
    run.tile_assembly = "mean"
    run.logger = None
    
    # Initialize the UNet model
    model = unet.UNet(
        n_input_channels=1,
        n_output_channels=1,
        encoder_channels=[64, 128, 256, 512, 1024],
        decoder_channels=[512, 256, 128, 64],
        device=get_device(),
    )
    #model.tile_size = 128
    #model.tile_assembly = "mean"
    model.x_coord = np.array([0, 128])
    model.y_coord = np.array([0, 128])
    model.nearest_map = None

    run.run()
    
    assert run.model_out_y_dir.exists()
    assert (run.model_out_y_dir / "test_pred.tif").exists()
