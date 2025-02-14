import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

def plot_colored_curves(x, y, cmap_name='coolwarm', title="", xlabel="", ylabel=""):
    """
    Plots a set of curves with a logarithmic color mapping.

    Parameters:
        x (array-like): The x-coordinates for the curves (1D array).
        y (array-like): The y-coordinates for the curves (2D array, each row is a curve).
        cmap_name (str): Name of the matplotlib colormap to use (default: 'coolwarm').
        title (str): Title of the plot (default: '').
        xlabel (str): Label for the x-axis (default: '').
        ylabel (str): Label for the y-axis (default: '').

    Returns:
        None
    """
    # Set up the logarithmic color mapping
    cmap = plt.cm.get_cmap(cmap_name)
    indices = np.linspace(1, y.shape[0], y.shape[0])  # Avoid log(0) by starting from 1
    log_indices = np.power(indices, 3)
    norm = Normalize(vmin=log_indices.min(), vmax=log_indices.max())

    # Create the plot
    plt.figure(figsize=(8, 5))
    for i, log_index in enumerate(log_indices):
        color = cmap(norm(log_index))
        plt.plot(x, y[i, :], color=color, alpha=0.6, lw=2)  # Increased line width

    # Add color bar for reference
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, label='Curve Index (Logarithmic Scaling)', cax=plt.axes([0.92, 0.1, 0.02, 0.8]))

    # Add labels and title (if provided)
    if title:
        plt.title(title)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)

    # Remove axis for a clean look
    plt.axis('off')

    # Show the plot
    plt.show()
