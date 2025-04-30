"""Test the plotting functions in the reports module."""


def test_plot_sabya() -> None:
    """Test the plot_sabya function."""
    import matplotlib.pyplot as plt
    import pandas as pd

    from neuro_morpho.reports.plots import plot_sabya

    # Create a sample DataFrame
    data = pd.DataFrame({"Group1": [1, 2, 3, 4, 5], "Group2": [2, 3, 4, 5, 6], "Group3": [3, 4, 5, 6, 7]})

    # Call the plot_sabya function
    fig = plot_sabya(data)

    # Check if the figure is created
    assert isinstance(fig, plt.Figure)
    plt.close(fig)  # Close the figure to free up memory
