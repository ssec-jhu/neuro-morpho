import importlib
import math
import os
from pathlib import Path

import numpy as np
import torch

from . import __project__


def get_device():
    # Optional: detect CI environment
    is_ci = os.getenv("CI") == "true"

    if not is_ci and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def find_package_location(package=__project__):
    return Path(importlib.util.find_spec(package).submodule_search_locations[0])


def find_repo_location(package=__project__):
    return Path(find_package_location(package) / os.pardir)


class TilesMixin:
    """Mixin class for tiling images and labels."""

    def __init__(self, tile_size: int = 512, image_size: tuple[int, int] = (3334, 3334), tile_assembly: str = "nn"):
        """Initialize the mixin.
        Args:
            tile_size (int, optional): The size of the tiles. Defaults to 512.
            image_size (tuple[int, int], optional): The size of the images. Defaults to (3334, 3334).   
        """
        self.tile_size = tile_size
        self.tile_assembly = tile_assembly


        n_x = math.ceil(image_size[0] / self.tile_size)
        X_coord = np.zeros(n_x, dtype=int)
        gap_x = math.floor((self.tile_size * n_x - image_size[0]) / (n_x - 1))
        gap_x_plus_one__amount = self.tile_size * n_x - image_size[0] - gap_x * (n_x - 1)
        for i in range(1, n_x):
            if i <= gap_x_plus_one__amount:
                X_coord[i] = int(X_coord[i - 1] + self.tile_size - (gap_x + 1))
            else:
                X_coord[i] = int(X_coord[i - 1] + self.tile_size - gap_x)
        n_y = math.ceil(image_size[1] / self.tile_size)
        Y_coord = np.zeros(n_y, dtype=int)
        gap_y = math.floor((self.tile_size * n_y - image_size[1]) / (n_y - 1))
        gap_y_plus_one__amount = self.tile_size * n_y - image_size[1] - gap_y * (n_y - 1)
        for i in range(1, n_y):
            if i <= gap_y_plus_one__amount:
                Y_coord[i] = int(Y_coord[i - 1] + self.tile_size - (gap_y + 1))
            else:
                Y_coord[i] = int(Y_coord[i - 1] + self.tile_size - gap_y)

        if self.tile_assembly == "nn":  # prepare nearest neighbor map
            X_Coord = np.tile(X_coord, n_y) + (self.tile_size - 1) / 2
            Y_Coord = np.repeat(Y_coord, n_x) + (self.tile_size - 1) / 2
            y_grid, x_grid = np.meshgrid(np.arange(image_size[0]), np.arange(image_size[1]), indexing="ij")
            y_grid = y_grid[..., np.newaxis]
            x_grid = x_grid[..., np.newaxis]
            distances = np.sqrt((x_grid - X_Coord) ** 2 + (y_grid - Y_Coord) ** 2)
            nearest_map = np.argmin(distances, axis=-1)
        else:
            nearest_map = None

        self.x_coord = X_coord
        self.y_coord = Y_coord
        self.nearest_map = nearest_map

    def tile_image(self, x: np.ndarray) -> np.ndarray:
        """Tile the image to the size T x T.

        Args:
            x: Image to be tiled in form of numpy array
        Returns:
            np.ndarray: The stack of tiles collected in raster order
        """
        tiles = list()
        n_y = len(self.y_coord)
        n_x = len(self.x_coord)
        for i in range(n_y):
            for j in range(n_x):
                tile = x[
                    self.y_coord[i] : (self.y_coord[i] + self.tile_size),
                    self.x_coord[j] : (self.x_coord[j] + self.tile_size),
                ]
                tiles.append(tile)

        tiles = np.stack(tiles)[:, :, :, np.newaxis]  # add channel dimension
        return tiles
