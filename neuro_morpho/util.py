import importlib
import os
from pathlib import Path

import torch

from . import __project__


def find_package_location(package=__project__):
    return Path(importlib.util.find_spec(package).submodule_search_locations[0])


def find_repo_location(package=__project__):
    return Path(find_package_location(package) / os.pardir)


def get_device():
    # Optional: detect CI environment
    is_ci = os.getenv("CI") == "true"

    if not is_ci and torch.backends.mps.is_available():
        return "mps"
    if not is_ci and torch.cuda.is_available():
        return "cuda"
    return "cpu"
