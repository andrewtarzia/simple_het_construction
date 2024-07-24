"""Script for visualising scoring function."""

import pathlib

import matplotlib.pyplot as plt
import numpy as np


def ff(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fitness function."""
    return x / 20 + y / 2


def matching_visualisation() -> None:
    """Visualise fitness function."""
    x_range = np.linspace(0.01, 180, 100)
    y_range = np.linspace(0.01, 5, 100)
    x_lbl = "angle residual"
    y_lbl = "distance residual"

    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(8, 5))

    grid_x, grid_y = np.meshgrid(x_range, y_range)

    grid_z = ff(grid_x, grid_y)
    cntr1 = ax.contourf(
        x_range,
        y_range,
        grid_z,
        cmap="RdBu_r",
        vmin=0,
    )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel(x_lbl, fontsize=16)
    ax.set_ylabel(y_lbl, fontsize=16)

    cbar = fig.colorbar(cntr1, ax=ax)
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label("residual", fontsize=16)

    fig.tight_layout()
    fig.savefig(
        pathlib.Path("/home/atarzia/workingspace/cpl/figures/")
        / "ff_param.png",
        dpi=360,
        bbox_inches="tight",
    )
    plt.close()


def main() -> None:
    """Run script."""
    matching_visualisation()


if __name__ == "__main__":
    main()
