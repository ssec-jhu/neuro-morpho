"""Analyzes and patches breaks in predicted binary images."""

import cv2
import numpy as np
from scipy.spatial.distance import cdist

MAX_FIXABLE_DISTANCE = 6  # Maximum distance to consider a break fixable


class BreaksAnalyzer:
    """Analyzes and patches breaks in predicted binary images.

    This class provides methods to identify and fix breaks in the dendrite
    segmentation of the predicted binary images.
    """

    def masked_max(self, image: np.ndarray, point: tuple[int, int], kernel: np.ndarray) -> tuple[int, int]:
        """Find the maximum value and its coordinate in a masked region of an image.

        Args:
            image (np.ndarray): The input image.
            point (tuple[int, int]): The center of the mask.
            kernel (np.ndarray): The mask to apply.

        Returns:
            tuple[int, int]: The coordinate of the maximum value.
        """
        assert kernel.shape == (3, 3), "Kernel must be 3x3"
        x, y = point
        h, w = image.shape
        max_val = -np.inf
        max_coord = (y, x)  # Default to center

        for dy in range(-1, 2):
            for dx in range(-1, 2):
                ky, kx = dy + 1, dx + 1  # index in kernel
                if kernel[ky, kx] == 1:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        val = image[ny, nx]
                        if val > max_val:
                            max_val = val
                            max_coord = (ny, nx)

        return max_coord

    def create_connecting_line(
        self,
        line_mask: np.ndarray,
        pt1: tuple[int, int],
        pt2: tuple[int, int],
        pred_bin_img: np.ndarray,
        pred_img: np.ndarray,
    ) -> bool:
        """Draw a line on the mask connecting two points.

        The line is drawn with respect to the predicted image, following the path
        of highest probability.

        Args:
            line_mask (np.ndarray): The mask to draw the line on.
            pt1 (tuple[int, int]): The starting point of the line.
            pt2 (tuple[int, int]): The ending point of the line.
            pred_bin_img (np.ndarray): The binary prediction image.
            pred_img (np.ndarray): The probability map prediction image.

        Returns:
            bool: True if the line was successfully connected, False otherwise.
        """
        # Find the direction from pt1 on main branch to pt2 on the branch being connected
        vector = (pt1[0] - pt2[0], pt1[1] - pt2[1])
        length_cntr = 0
        line_connected_flag = False
        while (
            not line_connected_flag and length_cntr < 2 * MAX_FIXABLE_DISTANCE
        ):  # Limit iterations to prevent infinite loop
            kernel = np.zeros((3, 3), dtype=np.uint8)  # Create the kernel for the current step
            # Check the 8 surrounding pixels (excluding the center pixel)
            for i in range(3):
                for j in range(3):
                    if i == 1 and j == 1:  # Skip the center pixel
                        continue
                    if (j - 1) * vector[0] + (i - 1) * vector[1] > 0:
                        # Check if the pixel is in the direction of the vector
                        if 0 <= pt2[1] + (i - 1) < line_mask.shape[0] and 0 <= pt2[0] + (j - 1) < line_mask.shape[1]:
                            # Check if the pixel is within bounds
                            if line_mask[pt2[1] + (i - 1), pt2[0] + (j - 1)] == 0:
                                # Check if the pixel is not yet in line_mask
                                kernel[i, j] = 1

            coord = self.masked_max(pred_img, pt2, kernel)

            if pred_bin_img[coord] == 255:  # If the pixel is white in the binary image
                line_connected_flag = True  # Stop if we reached the white pixel
                continue

            # Add the point to the mask and advance the counter
            line_mask[coord] = 255
            pt2 = (coord[1], coord[0])
            vector = (pt1[0] - pt2[0], pt1[1] - pt2[1])
            length_cntr += 1

        return line_connected_flag

    def analyze_breaks(self, pred_bin_img: np.ndarray, pred_img: np.ndarray) -> np.ndarray:
        """Find and patch potential breaks in the predicted binary image.

        Args:
            pred_bin_img (np.ndarray): The binary prediction image.
            pred_img (np.ndarray): The probability map prediction image.

        Returns:
            np.ndarray: The patched binary image.
        """
        if pred_img is None:
            raise ValueError("Predicted image must be provided.")
        if pred_bin_img is None:
            raise ValueError("Predicted binary binary image must be provided.")

        pred_bin_fixed_img = pred_bin_img.copy()

        # Label connected components
        num_labels, labels = cv2.connectedComponents(pred_bin_fixed_img)

        # Store pixel coordinates of each component (excluding background label 0)
        components = {}
        for label in range(1, num_labels):  # Ignore label 0 (background)
            yx_coords = np.column_stack(np.where(labels == label))  # (row, col) -> (y, x)
            components[label] = yx_coords

        # Sort components by the number of pixels (in descending order) to get the biggest one
        components = sorted(components.items(), key=lambda item: len(item[1]), reverse=True)
        distances = []
        for i, (label, coords) in enumerate(components):
            if i == 0:
                continue  # Skip the first (biggest) component
            min_distance = np.inf
            closest_pair = None  # Store pixel pair (p1, p2)
            # Compute pairwise distances between pixels of component i and j
            dist_matrix = cdist(components[0][1], components[i][1])
            # Find the minimum distance in this pair
            min_idx = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
            dist_value = dist_matrix[min_idx]
            # Update global minimum distance and pixel pair
            if dist_value < min_distance:
                min_distance = dist_value
                closest_pair = (
                    tuple(components[0][1][min_idx[0]]),  # Closest pixel in component 0
                    tuple(components[i][1][min_idx[1]]),  # Closest pixel in component i
                )
            distances.append((i, label, min_distance, len(coords), closest_pair))  # Save results
        distances.sort(key=lambda x: (x[2], -x[3]))  # sort by distance, then by size in descending order
        if len(distances) == 0:  # No breaks found in the predicted binary image.
            return pred_bin_fixed_img

        shortest_distance = distances[0][2]
        added_coords = None
        while len(distances) >= 1:  # Let's add components to the biggest one until there is only one left
            distance2CurrentComp = distances[0][2]
            if distance2CurrentComp > shortest_distance:  # Recalculate distances from biggest component to others
                for i, (comp_indx, label, min_dist, size, pair) in enumerate(distances):
                    closest_pair = None
                    comp_indx = distances[i][0]
                    dist_matrix = cdist(added_coords, components[comp_indx][1])
                    min_idx = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
                    dist_value = dist_matrix[min_idx]
                    if dist_value < distances[i][2]:
                        old = distances[i]
                        new = (
                            old[0],
                            old[1],
                            dist_value,
                            old[3],
                            (tuple(added_coords[min_idx[0]]), tuple(components[comp_indx][1][min_idx[1]])),
                        )
                        distances[i] = new

                # Resort distances after recalculating
                distances.sort(key=lambda x: (x[2], -x[3]))  # sort by distance
                shortest_distance = distances[0][2]
                distance2CurrentComp = distances[0][2]
                added_coords = None  # Reset added_coords

            if distance2CurrentComp >= MAX_FIXABLE_DISTANCE:  # If the distance is too big, stop merging
                break
            min_dist_indx = distances[0][0]
            closest_pair = distances[0][4]

            # Draw line on a temp image and get white pixel coords
            line_mask = np.zeros_like(pred_bin_fixed_img)
            pt1 = closest_pair[0][::-1]  # (x, y)
            pt2 = closest_pair[1][::-1]
            line_connected_flag = self.create_connecting_line(line_mask, pt1, pt2, pred_bin_fixed_img, pred_img)
            if not line_connected_flag:
                cv2.line(line_mask, pt1, pt2, color=255, thickness=1)
            pred_bin_fixed_img = cv2.bitwise_or(
                pred_bin_fixed_img, line_mask
            )  # Add the connection to the output binary image

            # Update the coordinates of main branch (sorted_components[0])
            line_coords = np.column_stack(np.where(line_mask == 255))  # (y, x)
            main_label = components[0][0]  # Get the label of the component you’re keeping
            merged_coords = np.vstack(
                (  # Merge coordinates: (line + component you're merging in)
                    components[0][1],  # Original coords of component 0
                    line_coords,  # Coords of the line connecting them
                    components[min_dist_indx][1],  # Coords of component min_dist_indx
                )
            )
            components[0] = (main_label, merged_coords)  # Update sorted_components[0] with the new merged component
            del distances[0]  # Update distances list
            # Create / update list of cooords for future recalculation of distances
            if added_coords is not None:
                added_coords = np.vstack(
                    (  # Add coordinates: (line + component you're merging in)
                        added_coords,  # Original coords of component 0
                        components[min_dist_indx][1],  # Coords of component min_dist_indx
                        line_coords,  # Coords of the line connecting them
                    )
                )
            else:
                added_coords = np.vstack(
                    (  # Add coordinates: (line + component you're merging in)
                        components[min_dist_indx][1],  # Coords of component min_dist_indx
                        line_coords,  # Coords of the line connecting them
                    )
                )

        return pred_bin_fixed_img
