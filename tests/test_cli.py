"""Unit tests for the neuro_morpho CLI."""

import gin
import gin.config
import pytest


from neuro_morpho import cli

def test_register_transforms():
    """Test that the torch transforms are registered correctly."""
    from torchvision.transforms.v2 import Compose, CenterCrop, RandomCrop, ToTensor, ToImage, ToDtype

    cli.register_torch_transforms()

    for transform in [Compose, CenterCrop, RandomCrop, ToTensor, ToImage, ToDtype]:
        assert gin.config._REGISTRY.get_match(transform.__name__) is not None, (
            f"{transform.__name__} is not registered in the gin config."
        )


def test_main_raises():
    """Test that the main function raises an error when no config is provided."""
    with pytest.raises(IOError, match="Unable to open") as exc_info:
        cli.main(config="does/not/exist.gin")


if __name__ == "__main__":
    # Run the test
    test_main_raises()