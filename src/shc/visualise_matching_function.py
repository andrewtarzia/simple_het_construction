"""Script for visualising scoring function."""

import itertools as it
import pathlib

import matplotlib.pyplot as plt
import numpy as np

from shc.definitions import EnvVariables
from shc.matching_functions import mismatch_test, plot_pair_position


def fake_mismatch_test() -> None:
    """Make a mismatch test."""
    figures_dir = pathlib.Path(
        "/home/atarzia/workingspace/cpl/figures/matching_test"
    )
    scores = []

    # A perfect one.
    c_dict1 = {
        "NN_distance": 10,
        "NN_BCN_angles": [90, 90],
    }
    c_dict2 = {
        "NN_distance": 10,
        "NN_BCN_angles": [90, 90],
    }
    pair_results = mismatch_test(c_dict1=c_dict1, c_dict2=c_dict2)

    scores.append((pair_results.state_1_result, pair_results.state_2_result))

    plot_pair_position(
        r1=np.array(
            (
                pair_results.set_parameters[0],
                pair_results.set_parameters[1],
            )
        ),
        phi1=pair_results.set_parameters[2],
        rigidbody1=pair_results.rigidbody1,
        r2=np.array(
            (
                pair_results.state_1_parameters[0],
                pair_results.state_1_parameters[1],
            )
        ),
        phi2=pair_results.state_1_parameters[2],
        rigidbody2=pair_results.rigidbody2,
        r3=np.array(
            (
                pair_results.state_2_parameters[0],
                pair_results.state_2_parameters[1],
            )
        ),
        phi3=pair_results.state_2_parameters[2],
        rigidbody3=pair_results.rigidbody3,
        outname=figures_dir / "v_-1.png",
    )

    nndist = 5
    nn_bcn_angles1 = np.linspace(70, 110, 2)
    nn_bcn_angles2 = np.linspace(150, 130, 2)

    for idx, iterate in enumerate(
        it.product(
            nn_bcn_angles2,
            nn_bcn_angles1,
            nn_bcn_angles1,
            nn_bcn_angles2,
        )
    ):
        c_dict1 = {
            "NN_distance": nndist,
            "NN_BCN_angles": [iterate[0], iterate[1] + 10],
        }
        c_dict2 = {
            "NN_distance": nndist,
            "NN_BCN_angles": [iterate[2], iterate[3]],
        }

        pair_results = mismatch_test(c_dict1=c_dict1, c_dict2=c_dict2)

        scores.append(
            (pair_results.state_1_result, pair_results.state_2_result)
        )

        plot_pair_position(
            r1=np.array(
                (
                    pair_results.set_parameters[0],
                    pair_results.set_parameters[1],
                )
            ),
            phi1=pair_results.set_parameters[2],
            rigidbody1=pair_results.rigidbody1,
            r2=np.array(
                (
                    pair_results.state_1_parameters[0],
                    pair_results.state_1_parameters[1],
                )
            ),
            phi2=pair_results.state_1_parameters[2],
            rigidbody2=pair_results.rigidbody2,
            r3=np.array(
                (
                    pair_results.state_2_parameters[0],
                    pair_results.state_2_parameters[1],
                )
            ),
            phi3=pair_results.state_2_parameters[2],
            rigidbody3=pair_results.rigidbody3,
            outname=figures_dir / f"v_{idx}.png",
        )

    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(8, 5))

    ax.plot(
        [i[0] for i in scores[1:]],
        c="tab:blue",
        marker="o",
        markersize=8,
        mec="k",
    )
    ax.plot(
        [i[1] for i in scores[1:]],
        c="tab:orange",
        marker="o",
        markersize=8,
        mec="k",
    )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel("idx", fontsize=16)
    ax.set_ylabel("score", fontsize=16)
    ax.set_yscale("log")

    fig.tight_layout()
    fig.savefig(
        pathlib.Path("/home/atarzia/workingspace/cpl/figures/matching_test")
        / "ascreen_score.png",
        dpi=360,
        bbox_inches="tight",
    )
    plt.close()


def ff(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fitness function."""
    # Times by two because we assume both sides have the same residual.
    x = 2 * ((0.5) * EnvVariables.k_angle * (np.radians(x)) ** 2)
    # Times by two because we assume both sides have the same residual.
    y = 2 * (0.5) * EnvVariables.k_bond * y**2

    return x + y


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
        levels=20,
    )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel(x_lbl, fontsize=16)
    ax.set_ylabel(y_lbl, fontsize=16)

    cbar = fig.colorbar(cntr1, ax=ax)
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label("residual", fontsize=16)

    fig.tight_layout()
    fig.savefig(
        pathlib.Path("/home/atarzia/workingspace/cpl/figures/matching_test")
        / "ff_param.png",
        dpi=360,
        bbox_inches="tight",
    )
    plt.close()


def main() -> None:
    """Run script."""
    matching_visualisation()
    fake_mismatch_test()


if __name__ == "__main__":
    main()
