import argparse

import gin
import gin.torch.external_configurables
from torchvision.transforms import v2

from neuro_morpho import run


def register_torch_transforms():
    """Register torch transforms to gin."""

    gin.external_configurable(v2.Compose, name="Compose")
    gin.external_configurable(v2.CenterCrop, name="CenterCrop")
    gin.external_configurable(v2.RandomCrop, name="RandomCrop")


if __name__ == "__main__":
    register_torch_transforms()

    parser = argparse.ArgumentParser(
        prog="neuro_morpho",
        description="NeuroMorpho training and reporting CLI.",
    )

    parser.add_argument(
        "config",
        nargs="?",
        type=str,
        help="The path to the gin configuration file.",
    )

    args = parser.parse_args()
    config = args.config or "config.gin"
    print(f"Loading gin config from {config}")
    gin.parse_config_file(config)
    run.run()
