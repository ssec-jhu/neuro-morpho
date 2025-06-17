import math

import numpy as np


class Tiler:
    """Helper class to tile images."""

    def __init__(
        self,
        image_size: int = (3334, 3334),
        tile_size: int = 512,
        tile_assembly: str = "nn",
    ):
        super().__init__()
        self.tile_size = tile_size
        self.tile_assembly = tile_assembly

        def get_tiling_attributes(image_size, tile_size, tile_assembly):
            """Create an instance of Tiler based on image size, tile size and tile assembly method."""
            n_x = math.ceil(image_size[0] / tile_size)
            x_coords = np.zeros(n_x, dtype=int)
            if n_x == 1:
                gap_x = 0
            else:
                gap_x = math.floor((tile_size * n_x - image_size[0]) / (n_x - 1))
            gap_x_plus_one__amount = tile_size * n_x - image_size[0] - gap_x * (n_x - 1)
            for i in range(1, n_x):
                if i <= gap_x_plus_one__amount:
                    x_coords[i] = int(x_coords[i - 1] + tile_size - (gap_x + 1))
                else:
                    x_coords[i] = int(x_coords[i - 1] + tile_size - gap_x)

            n_y = math.ceil(image_size[1] / tile_size)
            y_coords = np.zeros(n_y, dtype=int)
            if n_y == 1:
                gap_y = 0
            else:
                gap_y = math.floor((tile_size * n_y - image_size[1]) / (n_y - 1))
            gap_y_plus_one__amount = tile_size * n_y - image_size[1] - gap_y * (n_y - 1)
            for i in range(1, n_y):
                if i <= gap_y_plus_one__amount:
                    y_coords[i] = int(y_coords[i - 1] + tile_size - (gap_y + 1))
                else:
                    y_coords[i] = int(y_coords[i - 1] + tile_size - gap_y)

            nearest_map = None
            if tile_assembly == "nn":  # prepare nearest neighbor map
                x_centers = np.tile(x_coords, n_y) + (tile_size - 1) / 2
                y_centers = np.repeat(y_coords, n_x) + (tile_size - 1) / 2
                y_grid, x_grid = np.meshgrid(np.arange(image_size[0]), np.arange(image_size[1]), indexing="ij")
                y_grid = y_grid[..., np.newaxis]
                x_grid = x_grid[..., np.newaxis]
                distances = np.sqrt((x_grid - x_centers) ** 2 + (y_grid - y_centers) ** 2)
                nearest_map = np.argmin(distances, axis=-1)

            return x_coords, x_coords, nearest_map

        self.x_coords, self.y_coords, self.nearest_map = get_tiling_attributes(image_size, tile_size, tile_assembly)

    def tile_image(self, x: np.ndarray) -> np.ndarray:
        """Tile the image to tile_size x tile_size pieces in raster order."""
        tiles = list()
        n_y = len(self.y_coords)
        n_x = len(self.x_coords)
        for i in range(n_y):
            for j in range(n_x):
                tile = x[
                    self.y_coords[i] : (self.y_coords[i] + self.tile_size),
                    self.x_coords[j] : (self.x_coords[j] + self.tile_size),
                ]
                tiles.append(tile)

        return np.stack(tiles)[:, :, :, np.newaxis]  # add channel dimension
