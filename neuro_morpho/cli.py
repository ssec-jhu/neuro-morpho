"""Command line interface for training and running models."""

import fire
import gin
import torchvision
import torchvision.transforms.v2

from neuro_morpho import run


def register_torch_transforms() -> None:
    """Register torch transforms to gin.

    This allows the user to configure torchvision transforms from the gin config file.
    """
    gin.external_configurable(torchvision.transforms.v2.Compose, module="torchvision.transforms.v2")
    gin.external_configurable(torchvision.transforms.v2.CenterCrop, module="torchvision.transforms.v2")
    gin.external_configurable(torchvision.transforms.v2.RandomCrop, module="torchvision.transforms.v2")
    gin.external_configurable(torchvision.transforms.v2.ToTensor, module="torchvision.transforms.v2")
    gin.external_configurable(torchvision.transforms.v2.ToImage, module="torchvision.transforms.v2")
    gin.external_configurable(torchvision.transforms.v2.ToDtype, module="torchvision.transforms.v2")


def main(config: str = "unet.config.gin") -> None:
    """Run the training and inference pipeline.

    Args:
        config (str): The path to the gin configuration file.
    """
    register_torch_transforms()
    gin.parse_config_file(config)
    run.run()


if __name__ == "__main__":
    fire.Fire(main)
