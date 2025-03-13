import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_sabya(
    data: pd.DataFrame,
    ax: plt.Axes | None = None,
    fig_kw: dict = {},
    bar_width: float = 0.5,
    ax_ylabel: str = "",
) -> plt.Figure:
    """Make the pseudo box plot that sabya uses.

    Trying to emulate the plots generated using:
    https://github.com/raacampbell/notBoxPlot

    From the README:

    his function (with a tongue in cheek name) addresses this
    problem. The use of the mean instead of the median and the SEM and SD
    instead of quartiles and whiskers are deliberate. Jittered raw data are
    plotted for each group. Also shown are the mean, and 95% confidence
    intervals for the mean.

    Args:
        data (pd.DataFrame): The data to plot
        ax (plt.Axes, optional): The axes to plot on. Defaults to None.
        fig_kw (dict, optional): The figure keyword arguments. Defaults to {}.
        bar_width (float, optional): The width of the bars. Defaults to 0.5.
        ax_ylabel (str, optional): The y-axis label. Defaults to "".

    Returns:
        plt.Figure: The figure object
    """

    arr = data.to_numpy()

    mean = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0)
    sem = std / np.sqrt(arr.shape[0])

    locs = np.arange(arr.shape[1])

    ax = ax or plt.subplots(**fig_kw)[1]
    ax.bar(
        locs,
        2 * std,
        bottom=mean - std,
        color="mediumpurple",
        width=bar_width,
    )

    ax.bar(
        locs,
        2 * sem,
        bottom=mean - sem,
        color="lightcoral",
        width=bar_width,
    )

    for i in locs:
        ax.hlines(
            mean[i],
            i - bar_width / 2,
            i + bar_width / 2,
            color="red",
            linewidth=1,
        )

        ax.scatter(
            np.random.uniform(i - bar_width / 2, i + bar_width / 2, size=arr.shape[0]),
            arr[:, i],
            color="k",
            alpha=0.2,
        )

    ax.set_ylim(
        np.nanmin(arr) - 2 * std.min(),
        np.nanmax(arr) + 2 * std.max(),
    )
    ax.set_ylabel(ax_ylabel)

    ax.set_xticks(locs)
    ax.set_xticklabels(data.columns)

    return ax.figure


if __name__ == "__main__":
    import pandas as pd

    data = pd.DataFrame(np.random.normal(5, 15, size=(100, 4)), columns=["a", "b", "c", "d"])

    fig = plot_sabya(data)

    plt.show()
