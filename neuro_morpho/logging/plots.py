"""Plots for logging"""

import matplotlib.pyplot as plt
import numpy as np


def plot_triplet(
    in_img: np.ndarray,
    lbl_img: np.ndarray,
    out_img: np.ndarray,
) -> plt.Figure:
    """Plot a triplet of images: input, predicted, and label.

    Args:
        in_img (np.ndarray): Input image.
        lbl_img (np.ndarray): Label image.
        out_img (np.ndarray): Predicted image.
    Returns:
        plt.Figure: Figure containing the triplet plot.
    """
    fig, (ax_x, ax_pred, ax_y) = plt.subplots(ncols=3, nrows=1, figsize=(30, 10))
    ax_x.imshow(np.log(in_img), cmap="Greys_r")
    ax_x.set_title("log(Input)")
    ax_x.axis("off")
    ax_pred.imshow(out_img, vmin=0, vmax=1, cmap="Greys_r")
    ax_pred.set_title("Predicted")
    ax_pred.axis("off")
    ax_y.imshow(lbl_img, cmap="Greys_r")
    ax_y.set_title("Label")
    ax_y.axis("off")

    return fig
