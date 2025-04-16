"""Module for matching functions."""

from collections import abc

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from scipy import optimize

from shc.definitions import MatchingSettings
from shc.geometry import (
    LhsRigidBody,
    Pair,
    PairResult,
    RhsRigidBody,
    RigidBody,
)


def angle_test(
    c_dict1: dict[str, float | tuple],
    c_dict2: dict[str, float | tuple],
) -> float:
    """Uses study_1 angle test.

    NOT REQUIRED WITH STUDY 2, but here for comparison.
    """
    # NOT REQUIRED WITH STUDY 2:
    # 180 - angle, to make it the angle toward the binding interaction.

    angles1_sum = sum(c_dict1["NN_BCN_angles"])
    angles2_sum = sum(c_dict2["NN_BCN_angles"])

    interior_angles = angles1_sum + angles2_sum
    return interior_angles / 360


def mismatch_test(
    c_dict1: dict[str, float | tuple],
    c_dict2: dict[str, float | tuple],
    k_bond: float,
    k_angle: float,
) -> PairResult:
    """Test mismatch."""
    rigidbody1 = LhsRigidBody(
        nn_length=c_dict1["NN_distance"],
        theta1=c_dict1["NN_BCN_angles"][0],
        theta2=c_dict1["NN_BCN_angles"][1],
        vector_length=MatchingSettings.vector_length,
    )

    rigidbody2 = RhsRigidBody(
        nn_length=c_dict2["NN_distance"],
        theta1=c_dict2["NN_BCN_angles"][0],
        theta2=c_dict2["NN_BCN_angles"][1],
        vector_length=MatchingSettings.vector_length,
    )

    rigidbody3 = RhsRigidBody(
        nn_length=c_dict2["NN_distance"],
        theta1=c_dict2["NN_BCN_angles"][1],
        theta2=c_dict2["NN_BCN_angles"][0],
        vector_length=MatchingSettings.vector_length,
    )

    # Opt state 1.
    def f(params: abc.Sequence[float]) -> float:
        r1x, r1y, phi1 = MatchingSettings.set_state
        r2x, r2y, phi2 = params
        return Pair(lhs=rigidbody1, rhs=rigidbody2).calculate_residual(
            r1=np.array((r1x, r1y)),
            phi1=phi1,
            r2=np.array((r2x, r2y)),
            phi2=phi2,
            k_bond=k_bond,
            k_angle=k_angle,
        )

    state_1_results = []
    state_1_sets = []
    for initial_guess in MatchingSettings.initial_guesses:
        result = optimize.minimize(
            f,
            initial_guess,
            method=MatchingSettings.method,
            bounds=MatchingSettings.bounds,
            # options={"gtol": 1e-10},  # noqa: ERA001
            # options={"disp": True},  # noqa: ERA001
        )

        state_1_sets.append(result.x)
        state_1_results.append(
            Pair(lhs=rigidbody1, rhs=rigidbody2).calculate_residual(
                r1=np.array(
                    (
                        MatchingSettings.set_state[0],
                        MatchingSettings.set_state[1],
                    )
                ),
                phi1=MatchingSettings.set_state[2],
                r2=np.array((result.x[0], result.x[1])),
                phi2=result.x[2],
                k_bond=k_bond,
                k_angle=k_angle,
            )
        )

    state_1_result = min(state_1_results)
    state_1_parameters = state_1_sets[state_1_results.index(state_1_result)]

    # Opt state 2.
    def f(params: abc.Sequence[float]) -> float:
        r1x, r1y, phi1 = MatchingSettings.set_state
        r2x, r2y, phi2 = params
        return Pair(lhs=rigidbody1, rhs=rigidbody3).calculate_residual(
            r1=np.array((r1x, r1y)),
            phi1=phi1,
            r2=np.array((r2x, r2y)),
            phi2=phi2,
            k_bond=k_bond,
            k_angle=k_angle,
        )

    state_2_results = []
    state_2_sets = []
    for initial_guess in MatchingSettings.initial_guesses:
        result = optimize.minimize(
            f,
            initial_guess,
            method=MatchingSettings.method,
            bounds=MatchingSettings.bounds,
            # options={"gtol": 1e-10},  # noqa: ERA001
            # options={"disp": True},  # noqa: ERA001
        )

        state_2_sets.append(result.x)
        state_2_results.append(
            Pair(lhs=rigidbody1, rhs=rigidbody3).calculate_residual(
                r1=np.array(
                    (
                        MatchingSettings.set_state[0],
                        MatchingSettings.set_state[1],
                    )
                ),
                phi1=MatchingSettings.set_state[2],
                r2=np.array((result.x[0], result.x[1])),
                phi2=result.x[2],
                k_bond=k_bond,
                k_angle=k_angle,
            )
        )

    state_2_result = min(state_2_results)

    state_2_parameters = state_2_sets[state_2_results.index(state_2_result)]
    return PairResult(
        rigidbody1=rigidbody1,
        rigidbody2=rigidbody2,
        rigidbody3=rigidbody3,
        state_1_result=state_1_result,
        state_2_result=state_2_result,
        state_1_parameters=state_1_parameters,
        state_2_parameters=state_2_parameters,
        set_parameters=MatchingSettings.set_state,
    )


def plot_pair_position(  # noqa: PLR0913
    r1: np.ndarray,
    phi1: float,
    rigidbody1: RigidBody,
    r2: np.ndarray,
    phi2: float,
    rigidbody2: RigidBody,
    r3: np.ndarray,
    phi3: float,
    rigidbody3: RigidBody,
    outname: str,
) -> None:
    """Plot pair rigid bodies."""
    fig, ax = plt.subplots(ncols=1, figsize=(8, 5))

    # Plot residuals.
    ax.plot(
        (rigidbody2.get_x1(r2, phi2).x, rigidbody1.get_x1(r1, phi1).x),
        (rigidbody2.get_x1(r2, phi2).y, rigidbody1.get_x1(r1, phi1).y),
        c="tab:red",
        alpha=1.0,
        lw=1,
        ls="--",
    )
    ax.plot(
        (rigidbody2.get_x2(r2, phi2).x, rigidbody1.get_x2(r1, phi1).x),
        (rigidbody2.get_x2(r2, phi2).y, rigidbody1.get_x2(r1, phi1).y),
        c="tab:red",
        alpha=1.0,
        lw=1,
        ls="--",
    )
    ax.plot(
        (rigidbody3.get_x1(r3, phi3).x + 8, rigidbody1.get_x1(r1, phi1).x + 8),
        (rigidbody3.get_x1(r3, phi3).y, rigidbody1.get_x1(r1, phi1).y),
        c="tab:green",
        alpha=1.0,
        lw=1,
        ls="--",
    )
    ax.plot(
        (rigidbody3.get_x2(r3, phi3).x + 8, rigidbody1.get_x2(r1, phi1).x + 8),
        (rigidbody3.get_x2(r3, phi3).y, rigidbody1.get_x2(r1, phi1).y),
        c="tab:green",
        alpha=1.0,
        lw=1,
        ls="--",
    )

    # Plot LHS ligand.
    ax.plot(
        (rigidbody1.get_n1(r1, phi1).x, rigidbody1.get_n2(r1, phi1).x),
        (rigidbody1.get_n1(r1, phi1).y, rigidbody1.get_n2(r1, phi1).y),
        c="k",
        alpha=1.0,
        lw=2,
        marker="o",
        markersize=6,
        markeredgecolor="none",
    )

    ax.plot(
        (rigidbody1.get_n1(r1, phi1).x + 8, rigidbody1.get_x1(r1, phi1).x + 8),
        (rigidbody1.get_n1(r1, phi1).y, rigidbody1.get_x1(r1, phi1).y),
        c="tab:orange",
        alpha=1.0,
        lw=2,
        marker="o",
        markersize=6,
        markeredgecolor="none",
    )

    ax.plot(
        (rigidbody1.get_n2(r1, phi1).x + 8, rigidbody1.get_x2(r1, phi1).x + 8),
        (rigidbody1.get_n2(r1, phi1).y, rigidbody1.get_x2(r1, phi1).y),
        c="tab:orange",
        alpha=1.0,
        lw=2,
        marker="o",
        markersize=6,
        markeredgecolor="none",
    )

    ax.plot(
        (rigidbody1.get_n1(r1, phi1).x + 8, rigidbody1.get_n2(r1, phi1).x + 8),
        (rigidbody1.get_n1(r1, phi1).y, rigidbody1.get_n2(r1, phi1).y),
        c="k",
        alpha=1.0,
        lw=2,
        marker="o",
        markersize=6,
        markeredgecolor="none",
    )

    ax.plot(
        (rigidbody1.get_n1(r1, phi1).x, rigidbody1.get_x1(r1, phi1).x),
        (rigidbody1.get_n1(r1, phi1).y, rigidbody1.get_x1(r1, phi1).y),
        c="tab:orange",
        alpha=1.0,
        lw=2,
        marker="o",
        markersize=6,
        markeredgecolor="none",
    )

    ax.plot(
        (rigidbody1.get_n2(r1, phi1).x, rigidbody1.get_x2(r1, phi1).x),
        (rigidbody1.get_n2(r1, phi1).y, rigidbody1.get_x2(r1, phi1).y),
        c="tab:orange",
        alpha=1.0,
        lw=2,
        marker="o",
        markersize=6,
        markeredgecolor="none",
    )

    # Plot RHS ligand.
    ax.plot(
        (rigidbody2.get_n1(r2, phi2).x, rigidbody2.get_n2(r2, phi2).x),
        (rigidbody2.get_n1(r2, phi2).y, rigidbody2.get_n2(r2, phi2).y),
        c="k",
        alpha=1.0,
        lw=2,
        marker="o",
        markersize=6,
        markeredgecolor="none",
    )

    ax.plot(
        (rigidbody2.get_n1(r2, phi2).x, rigidbody2.get_x1(r2, phi2).x),
        (rigidbody2.get_n1(r2, phi2).y, rigidbody2.get_x1(r2, phi2).y),
        c="tab:cyan",
        alpha=1.0,
        lw=2,
        marker="o",
        markersize=6,
        markeredgecolor="none",
    )

    ax.plot(
        (rigidbody2.get_n2(r2, phi2).x, rigidbody2.get_x2(r2, phi2).x),
        (rigidbody2.get_n2(r2, phi2).y, rigidbody2.get_x2(r2, phi2).y),
        c="tab:cyan",
        alpha=1.0,
        lw=2,
        marker="o",
        markersize=6,
        markeredgecolor="none",
    )

    # Plot RHS ligand state 2.
    ax.plot(
        (rigidbody3.get_n1(r3, phi3).x + 8, rigidbody3.get_n2(r3, phi3).x + 8),
        (rigidbody3.get_n1(r3, phi3).y, rigidbody3.get_n2(r3, phi3).y),
        c="k",
        alpha=1.0,
        lw=2,
        marker="o",
        markersize=6,
        markeredgecolor="none",
    )
    ax.plot(
        (rigidbody3.get_n1(r3, phi3).x + 8, rigidbody3.get_x1(r3, phi3).x + 8),
        (rigidbody3.get_n1(r3, phi3).y, rigidbody3.get_x1(r3, phi3).y),
        c="tab:cyan",
        alpha=1.0,
        lw=2,
        marker="o",
        markersize=6,
        markeredgecolor="none",
    )

    ax.plot(
        (rigidbody3.get_n2(r3, phi3).x + 8, rigidbody3.get_x2(r3, phi3).x + 8),
        (rigidbody3.get_n2(r3, phi3).y, rigidbody3.get_x2(r3, phi3).y),
        c="tab:cyan",
        alpha=1.0,
        lw=2,
        marker="o",
        markersize=6,
        markeredgecolor="none",
    )

    circ = patches.Circle(
        (rigidbody1.get_n1(r1, phi1).x, rigidbody1.get_n1(r1, phi1).y),
        2.02,
        alpha=0.2,
        fc="tab:orange",
    )
    ax.add_patch(circ)
    circ = patches.Circle(
        (rigidbody1.get_n2(r1, phi1).x + 8, rigidbody1.get_n2(r1, phi1).y),
        2.02,
        alpha=0.2,
        fc="tab:orange",
    )
    ax.add_patch(circ)
    circ = patches.Circle(
        (rigidbody1.get_n1(r1, phi1).x + 8, rigidbody1.get_n1(r1, phi1).y),
        2.02,
        alpha=0.2,
        fc="tab:orange",
    )
    ax.add_patch(circ)
    circ = patches.Circle(
        (rigidbody1.get_n2(r1, phi1).x, rigidbody1.get_n2(r1, phi1).y),
        2.02,
        alpha=0.2,
        fc="tab:orange",
    )
    ax.add_patch(circ)
    circ = patches.Circle(
        (rigidbody2.get_n1(r2, phi2).x, rigidbody2.get_n1(r2, phi2).y),
        2.02,
        alpha=0.2,
        fc="tab:cyan",
    )
    ax.add_patch(circ)
    circ = patches.Circle(
        (rigidbody2.get_n2(r2, phi2).x, rigidbody2.get_n2(r2, phi2).y),
        2.02,
        alpha=0.2,
        fc="tab:cyan",
    )
    ax.add_patch(circ)
    circ = patches.Circle(
        (rigidbody3.get_n1(r3, phi3).x + 8, rigidbody3.get_n1(r3, phi3).y),
        2.02,
        alpha=0.2,
        fc="tab:cyan",
    )
    ax.add_patch(circ)
    circ = patches.Circle(
        (rigidbody3.get_n2(r3, phi3).x + 8, rigidbody3.get_n2(r3, phi3).y),
        2.02,
        alpha=0.2,
        fc="tab:cyan",
    )
    ax.add_patch(circ)

    ax.scatter(
        r1[0],
        r1[1],
        c="k",
        alpha=1.0,
        s=120,
        edgecolor="w",
        zorder=4,
    )
    ax.scatter(
        r1[0] + 8,
        r1[1],
        c="k",
        alpha=1.0,
        s=120,
        edgecolor="w",
        zorder=4,
    )
    ax.scatter(
        r2[0],
        r2[1],
        c="k",
        alpha=1.0,
        s=120,
        edgecolor="w",
        zorder=4,
    )
    ax.scatter(
        r3[0] + 8,
        r3[1],
        c="k",
        alpha=1.0,
        s=120,
        edgecolor="w",
        zorder=4,
    )

    lim = 16
    ratio = 8 / 5
    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.axis("off")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim / ratio, lim / ratio)
    ax.set_xticks([])
    ax.set_yticks([])

    fig.tight_layout()
    fig.savefig(
        f"{outname}",
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()
