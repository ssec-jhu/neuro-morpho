import argparse

import gin
import gin.torch.external_configurables
import torchvision
import torchvision.transforms.v2

from neuro_morpho import run


def register_torch_transforms():
    """Register torch transforms to gin."""
    gin.external_configurable(torchvision.transforms.v2.Compose, module="torchvision.transforms.v2")
    gin.external_configurable(torchvision.transforms.v2.CenterCrop, module="torchvision.transforms.v2")
    gin.external_configurable(torchvision.transforms.v2.RandomCrop, module="torchvision.transforms.v2")
    gin.external_configurable(torchvision.transforms.v2.ToTensor, module="torchvision.transforms.v2")


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
