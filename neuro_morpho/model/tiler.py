"""Image tiling and stitching."""

import math

import numpy as np


class Tiler:
    """Image tiling and stitching.

    This class provides methods for tiling an image into smaller patches and
    stitching them back together.
    """

    def __init__(
        self,
        tile_size: int = 512,
        tile_assembly: str = "nn",
        x_coords: np.ndarray | None = None,
        y_coords: np.ndarray | None = None,
        nearest_map: np.ndarray | None = None,
    ):
        """Initialize the Tiler.

        Args:
            tile_size (int, optional): The size of the tiles. Defaults to 512.
            tile_assembly (str, optional): The method for assembling the tiles.
                Can be 'nn' (nearest neighbor), 'mean', or 'max'. Defaults to "nn".
            x_coords (np.ndarray, optional): The x-coordinates of the tiles. Defaults to None.
            y_coords (np.ndarray, optional): The y-coordinates of the tiles. Defaults to None.
            nearest_map (np.ndarray, optional): The nearest neighbor map. Defaults to None.
        """
        super().__init__()
        self.tile_size = tile_size
        self.tile_assembly = tile_assembly

    def get_tiling_attributes(self, image_size):
        """Calculate the tiling attributes based on the image size.

        This method calculates the x and y coordinates of the tiles and the
        nearest neighbor map if the tile assembly method is 'nn'.

        Args:
            image_size (tuple[int, int]): The size of the image.
        """
        n_x = math.ceil(image_size[0] / self.tile_size)
        self.x_coords = np.zeros(n_x, dtype=int)
        if n_x == 1:
            gap_x = 0
        else:
            gap_x = math.floor((self.tile_size * n_x - image_size[0]) / (n_x - 1))
        gap_x_plus_one__amount = self.tile_size * n_x - image_size[0] - gap_x * (n_x - 1)
        for i in range(1, n_x):
            if i <= gap_x_plus_one__amount:
                self.x_coords[i] = int(self.x_coords[i - 1] + self.tile_size - (gap_x + 1))
            else:
                self.x_coords[i] = int(self.x_coords[i - 1] + self.tile_size - gap_x)
        n_y = math.ceil(image_size[1] / self.tile_size)
        self.y_coords = np.zeros(n_y, dtype=int)
        if n_y == 1:
            gap_y = 0
        else:
            gap_y = math.floor((self.tile_size * n_y - image_size[1]) / (n_y - 1))
        gap_y_plus_one__amount = self.tile_size * n_y - image_size[1] - gap_y * (n_y - 1)
        for i in range(1, n_y):
            if i <= gap_y_plus_one__amount:
                self.y_coords[i] = int(self.y_coords[i - 1] + self.tile_size - (gap_y + 1))
            else:
                self.y_coords[i] = int(self.y_coords[i - 1] + self.tile_size - gap_y)
        self.nearest_map = None
        if self.tile_assembly == "nn":  # prepare nearest neighbor map
            x_centers = np.tile(self.x_coords, n_y) + (self.tile_size - 1) / 2
            y_centers = np.repeat(self.y_coords, n_x) + (self.tile_size - 1) / 2
            y_grid, x_grid = np.meshgrid(np.arange(image_size[0]), np.arange(image_size[1]), indexing="ij")
            y_grid = y_grid[..., np.newaxis]
            x_grid = x_grid[..., np.newaxis]
            distances = np.sqrt((x_grid - x_centers) ** 2 + (y_grid - y_centers) ** 2)
            self.nearest_map = np.argmin(distances, axis=-1)

    def tile_image(self, image: np.ndarray) -> np.ndarray:
        """Tile an image into smaller patches.

        Args:
            image (np.ndarray): The input image.

        Returns:
            np.ndarray: An array of tiles.
        """
        y_bounds = [(y, y + self.tile_size) for y in self.y_coords]
        x_bounds = [(x, x + self.tile_size) for x in self.x_coords]
        tiles = [image[y0:y1, x0:x1] for y0, y1 in y_bounds for x0, x1 in x_bounds]

        return np.stack(tiles)
