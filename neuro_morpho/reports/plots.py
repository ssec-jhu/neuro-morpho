import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_sabya(
    data: pd.DataFrame,
    title: str,
    ax: plt.Axes | None = None,
    fig_kw: dict = {},
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
    """

    arr = data.to_numpy()

    mean = np.mean(arr, axis=0)
    std = np.std(arr, axis=0)
    sem = std / np.sqrt(arr.shape[0])

    sorted_arr = np.sort(arr, axis=0)
    mn = (np.less(sorted_arr, (mean - std)).sum(axis=0) / arr.shape[0]) * 100
    mx = (np.greater(sorted_arr, (mean + std)).sum(axis=0) / arr.shape[0]) * 100

    ax = ax or plt.subplots(**fig_kw)[1]
    ax.set_title(title)

    for i in range(arr.shape[1]):
        ax.boxplot(
            arr[:, i],
            positions=[i],
            usermedians=[mean[i]],
            conf_intervals=np.array([[mean[i] - sem[i], mean[i] + sem[i]]]),
            whis=[mn[i], mx[i]],
            tick_labels=[data.columns[i]],
            showfliers=False,
            medianprops={"color": "k"},
            boxprops={"color": "k"},
            whiskerprops={"color": "k"},
            widths=0.3,
        )

        locs = np.random.uniform(i - 0.15, i + 0.15, size=arr.shape[0])
        ax.plot(
            locs,
            arr[:, i],
            "o",
            color="k",
            alpha=0.2,
        )

    return ax.figure


if __name__ == "__main__":
    import pandas as pd

    csv_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    iris = pd.read_csv(csv_url, names=["sepal_length", "sepal_width", "petal_length", "petal_width", "class"])
    iris = iris.drop(columns="class")

    fig = plot_sabya(iris, title="Iris Dataset")

    plt.show()
