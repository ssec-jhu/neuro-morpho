import fire
import gin
import torchvision
import torchvision.transforms.v2

from neuro_morpho import run

# Register constants (only if they’re not already defined)
try:
    gin.constant("train_flag", False)
except ValueError:
    pass

try:
    gin.constant("test_flag", True)
except ValueError:
    pass


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
    # Query constants
    train = gin.query_parameter("train_flag")
    test = gin.query_parameter("test_flag")
    if train:
        run.train()
    if test:
        run.test()


if __name__ == "__main__":
    fire.Fire(main)
