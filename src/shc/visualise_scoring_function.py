"""Script for visualising scoring function."""

import pathlib

import matplotlib.pyplot as plt
import numpy as np


def scoring_function(x: np.ndarray, target: float, beta: float) -> np.ndarray:
    """Define ff."""
    return 1 / (1 + np.exp(beta * (x - target)))
    return 20 * np.exp(-1 * x)

    return 10 * np.exp(-1 * x)
    return 20 * np.exp(-1 * x)


def visualisation() -> None:
    """Visualise ff."""
    x_range = np.linspace(0.01, 5, 100)
    x_target = 2
    x_lbl = "1-residual"
    y_lbl = "$c$ 1-residual"

    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(8, 5))

    yvalues = scoring_function(x_range, x_target, 10.0)

    ax.plot(x_range, yvalues, c="tab:blue")

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel(x_lbl, fontsize=16)
    ax.set_ylabel(y_lbl, fontsize=16)
    ax.axvline(x=x_target, c="k")

    fig.tight_layout()
    fig.savefig(
        pathlib.Path("/home/atarzia/workingspace/cpl/figures/cs1/")
        / "ff_param.png",
        dpi=360,
        bbox_inches="tight",
    )
    plt.close()


def main() -> None:
    """Run script."""
    visualisation()


if __name__ == "__main__":
    main()
