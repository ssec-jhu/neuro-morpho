import fire
import comet_ml # importing this first to get all data logged automatically
import gin
import torchvision
import torchvision.transforms.v2

from neuro_morpho import run


def register_torch_transforms():
    """Register torch transforms to gin."""
    gin.external_configurable(torchvision.transforms.v2.Compose, module="torchvision.transforms.v2")
    gin.external_configurable(torchvision.transforms.v2.CenterCrop, module="torchvision.transforms.v2")
    gin.external_configurable(torchvision.transforms.v2.RandomCrop, module="torchvision.transforms.v2")
    gin.external_configurable(torchvision.transforms.v2.ToTensor, module="torchvision.transforms.v2")
    gin.external_configurable(torchvision.transforms.v2.ToImage, module="torchvision.transforms.v2")
    gin.external_configurable(torchvision.transforms.v2.ToDtype, module="torchvision.transforms.v2")


def main(config: str = "unet.config.gin") -> None:
    """Run the main function.

    Args:
        config (str): The path to the gin configuration file.
    """
    register_torch_transforms()
    gin.parse_config_file(config)
    run.run()


if __name__ == "__main__":
    fire.Fire(main)
