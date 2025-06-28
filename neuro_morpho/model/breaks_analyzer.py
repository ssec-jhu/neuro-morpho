import cv2
import numpy as np
from scipy.spatial.distance import cdist


class BreaksAnalyzer:
    """Helper class to analyze and patch the breaks in predicted binary images."""

    def __init__(
        self,
    ):
        super().__init__()

    def masked_max(image, point, kernel):
        # kernel should be a 3x3 array of 0s and 1s
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

        return max_val, max_coord

    def create_connecting_line(self, line_mask, pt1, pt2, pred_bin_img, pred_img):
        """Draw a line on the mask wrt predicted_image."""
        # Find the direction from pt1 on main branch to pt2 on the branch being connected
        vector = (pt1[0] - pt2[0], pt1[1] - pt2[1])
        while True:
            # Create the kernel for the current step
            kernel = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.uint8)
            for i in range(3):
                for j in range(3):
                    if i == 1 and j == 1:  # Skip the center pixel
                        continue
                    if (j - 1) * vector[0] + (i - 1) * vector[
                        1
                    ] > 0:  # Check if the pixel is in the direction of the vector
                        if (
                            0 <= pt2[1] + (i - 1) < line_mask.shape[0] and 0 <= pt2[0] + (j - 1) < line_mask.shape[1]
                        ):  # Check if the pixel is within bounds
                            if (
                                line_mask[pt2[1] + (i - 1), pt2[0] + (j - 1)] == 0
                            ):  # Check if the pixel is not yet in line_mask
                                kernel[i, j] = 1

            val, coord = self.masked_max(pred_img, pt2, kernel=kernel)

            if pred_bin_img[coord] == 255:  # If the pixel is white in the binary image
                break
            # Add the point to the mask
            line_mask[coord] = 255
            pt2 = (coord[1], coord[0])
            vector = (pt1[0] - pt2[0], pt1[1] - pt2[1])

        return True

    def analyze_breaks(self, pred_bin_img: np.ndarray, pred_img: np.ndarray) -> np.ndarray:
        """Find the potential break places in hard prediction (dendrite binary image) and patch it."""
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

            if distance2CurrentComp >= 6:  # If the distance is too big, stop merging
                break
            min_dist_indx = distances[0][0]
            closest_pair = distances[0][4]
            # print(
            #     f"Distance: {distance2CurrentComp:.2f}\t Size: {distances[0][3]}\t Closest pair: {closest_pair[0]} ↔ \
            #         {closest_pair[1]}"
            # )

            # output_image = cv2.cvtColor(pred_bin_fixed_img, cv2.COLOR_GRAY2BGR)
            # for y, x in components[0][1]:
            #     output_image[y, x] = [0, 255, 0]  # Green (BGR) - main branch
            # for y, x in components[min_dist_indx][1]:
            #     output_image[y, x] = [0, 0, 255]  # Red (BGR) - branch being connected
            # cv2.imshow(os.path.basename(binary_filename), output_image)
            # cv2.waitKey(0)

            # Draw line on a temp image and get white pixel coords
            line_mask = np.zeros_like(pred_bin_fixed_img)
            pt1 = closest_pair[0][::-1]  # (x, y)
            pt2 = closest_pair[1][::-1]
            result = self.create_connecting_line(line_mask, pt1, pt2, pred_bin_fixed_img, pred_img)
            if not result:
                cv2.line(line_mask, pt1, pt2, color=255, thickness=1)
            pred_bin_fixed_img = cv2.bitwise_or(
                pred_bin_fixed_img, line_mask
            )  # Add the connection to the output binary imag

            # for y, x in zip(*line_mask.nonzero()):
            #     output_image[y, x] = [255, 0, 0]  # Blue (BGR) - line connecting them
            # all_coords = np.vstack((components[min_dist_indx][1], np.column_stack(line_mask.nonzero())))
            # min_y, min_x = np.min(all_coords, axis=0)
            # max_y, max_x = np.max(all_coords, axis=0)
            # pad = 20
            # min_y = max(min_y - pad, 0)
            # min_x = max(min_x - pad, 0)
            # max_y = min(max_y + pad, output_image.shape[0] - 1)
            # max_x = min(max_x + pad, output_image.shape[1] - 1)
            # cv2.imshow(os.path.basename(binary_filename), output_image[min_y:max_y+1, min_x:max_x+1])
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

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

        # target_image = cv2.imread(target_filename, cv2.IMREAD_GRAYSCALE)
        # # diff_target = np.stack([pred_bin_fixed_img] * 3, axis=-1).astype(np.uint8)
        # diff_target[(pred_bin_fixed_img == 255) & (target_image == 0)] = \
        #     [0, 255, 0]  # Green: appears in fixed, missing in target
        # diff_target[(pred_bin_fixed_img == 0) & (target_image == 255)] = \
        #     [255, 0, 0]  # Red: missing in fixed, appears in target
        # diff_target_filename = binary_filename.replace("_pred_bin.tif", "_pred_bin_fixedimp_diff_tar.tif")
        # cv2.imwrite(diff_target_filename, cv2.cvtColor(diff_target, cv2.COLOR_RGB2BGR))
        #
        # diff_bin = np.stack([pred_bin_fixed_img] * 3, axis=-1).astype(np.uint8)
        # diff_bin[(pred_bin_fixed_img == 255) & (pred_bin_img == 0)] = \
        #     [0, 255, 0]  # Green: appears in fixed, missing in orig binary
        # diff_bin[(pred_bin_fixed_img == 0) & (pred_bin_img == 255)] = \
        #     [255, 0, 0]  # Red: missing in fixed, appears in orig binary
        # diff_bin_filename = binary_filename.replace("_pred_bin.tif", "_pred_bin_fixedimp_diff_bin.tif")
        # cv2.imwrite(diff_bin_filename, cv2.cvtColor(diff_bin, cv2.COLOR_RGB2BGR))

        return pred_bin_fixed_img
