"""Module for matching functions."""

import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
import matplotlib.patches as patches


def vector_length():
    """
    Mean value of bond distance to use in candidate selection.

    """
    return 2.02


def angle_test(c_dict1, c_dict2):
    # NOT REQUIRED WITH STUDY 2:
    # 180 - angle, to make it the angle toward the binding interaction.

    angles1_sum = sum(c_dict1["NN_BCN_angles"])
    angles2_sum = sum(c_dict2["NN_BCN_angles"])

    interior_angles = angles1_sum + angles2_sum
    return interior_angles / 360


def plot_pair_position(
    r1,
    phi1,
    rigidbody1,
    r2,
    phi2,
    rigidbody2,
    r3,
    phi3,
    rigidbody3,
    outname,
):
    fig, ax = plt.subplots(figsize=(4, 4))

    # Plot residuals.
    ax.plot(
        (rigidbody2.get_x1(r2).x, rigidbody1.get_x1(r1).x),
        (rigidbody2.get_x1(r2).y, rigidbody1.get_x1(r1).y),
        c="tab:red",
        alpha=1.0,
        # s=100,
        lw=1,
        ls="--",
    )
    ax.plot(
        (rigidbody2.get_x2(r2).x, rigidbody1.get_x2(r1).x),
        (rigidbody2.get_x2(r2).y, rigidbody1.get_x2(r1).y),
        c="tab:red",
        alpha=1.0,
        # s=100,
        lw=1,
        ls="--",
    )
    ax.plot(
        (rigidbody3.get_x1(r3).x, rigidbody1.get_x1(r1).x),
        (rigidbody3.get_x1(r3).y, rigidbody1.get_x1(r1).y),
        c="tab:green",
        alpha=1.0,
        # s=100,
        lw=1,
        ls="--",
    )
    ax.plot(
        (rigidbody3.get_x2(r3).x, rigidbody1.get_x2(r1).x),
        (rigidbody3.get_x2(r3).y, rigidbody1.get_x2(r1).y),
        c="tab:green",
        alpha=1.0,
        # s=100,
        lw=1,
        ls="--",
    )

    # Plot LHS ligand.
    ax.plot(
        (rigidbody1.get_n1(r1).x, rigidbody1.get_n2(r1).x),
        (rigidbody1.get_n1(r1).y, rigidbody1.get_n2(r1).y),
        c="tab:blue",
        alpha=1.0,
        # s=100,
        lw=2,
        marker="o",
        markersize=6,
        markeredgecolor="none",
    )

    ax.plot(
        (rigidbody1.get_n1(r1).x, rigidbody1.get_x1(r1).x),
        (rigidbody1.get_n1(r1).y, rigidbody1.get_x1(r1).y),
        c="tab:orange",
        alpha=1.0,
        # s=100,
        lw=2,
        marker="o",
        markersize=6,
        markeredgecolor="none",
    )

    ax.plot(
        (rigidbody1.get_n2(r1).x, rigidbody1.get_x2(r1).x),
        (rigidbody1.get_n2(r1).y, rigidbody1.get_x2(r1).y),
        c="tab:orange",
        alpha=1.0,
        # s=100,
        lw=2,
        marker="o",
        markersize=6,
        markeredgecolor="none",
    )

    # Plot RHS ligand.
    ax.plot(
        (rigidbody2.get_n1(r2).x, rigidbody2.get_n2(r2).x),
        (rigidbody2.get_n1(r2).y, rigidbody2.get_n2(r2).y),
        c="tab:blue",
        alpha=1.0,
        # s=100,
        lw=2,
        marker="o",
        markersize=6,
        markeredgecolor="none",
    )

    ax.plot(
        (rigidbody2.get_n1(r2).x, rigidbody2.get_x1(r2).x),
        (rigidbody2.get_n1(r2).y, rigidbody2.get_x1(r2).y),
        c="tab:orange",
        alpha=1.0,
        # s=100,
        lw=2,
        marker="o",
        markersize=6,
        markeredgecolor="none",
    )

    ax.plot(
        (rigidbody2.get_n2(r2).x, rigidbody2.get_x2(r2).x),
        (rigidbody2.get_n2(r2).y, rigidbody2.get_x2(r2).y),
        c="tab:orange",
        alpha=1.0,
        # s=100,
        lw=2,
        marker="o",
        markersize=6,
        markeredgecolor="none",
    )

    # Plot RHS ligand state 2.
    ax.plot(
        (rigidbody3.get_n1(r3).x, rigidbody3.get_n2(r3).x),
        (rigidbody3.get_n1(r3).y, rigidbody3.get_n2(r3).y),
        c="tab:cyan",
        alpha=1.0,
        # s=100,
        lw=2,
        marker="o",
        markersize=6,
        markeredgecolor="none",
    )
    ax.plot(
        (rigidbody3.get_n1(r3).x, rigidbody3.get_x1(r3).x),
        (rigidbody3.get_n1(r3).y, rigidbody3.get_x1(r3).y),
        c="tab:purple",
        alpha=1.0,
        # s=100,
        lw=2,
        marker="o",
        markersize=6,
        markeredgecolor="none",
    )

    ax.plot(
        (rigidbody3.get_n2(r3).x, rigidbody3.get_x2(r3).x),
        (rigidbody3.get_n2(r3).y, rigidbody3.get_x2(r3).y),
        c="tab:purple",
        alpha=1.0,
        # s=100,
        lw=2,
        marker="o",
        markersize=6,
        markeredgecolor="none",
    )

    circ = patches.Circle(
        (rigidbody1.get_n1(r1).x, rigidbody1.get_n1(r1).y),
        2.02,
        alpha=0.2,
        fc="yellow",
    )
    ax.add_patch(circ)
    circ = patches.Circle(
        (rigidbody1.get_n2(r1).x, rigidbody1.get_n2(r1).y),
        2.02,
        alpha=0.2,
        fc="yellow",
    )
    ax.add_patch(circ)
    circ = patches.Circle(
        (rigidbody2.get_n1(r2).x, rigidbody2.get_n1(r2).y),
        2.02,
        alpha=0.2,
        fc="yellow",
    )
    ax.add_patch(circ)
    circ = patches.Circle(
        (rigidbody2.get_n2(r2).x, rigidbody2.get_n2(r2).y),
        2.02,
        alpha=0.2,
        fc="yellow",
    )
    ax.add_patch(circ)
    circ = patches.Circle(
        (rigidbody3.get_n1(r3).x, rigidbody3.get_n1(r3).y),
        2.02,
        alpha=0.2,
        fc="yellow",
    )
    ax.add_patch(circ)
    circ = patches.Circle(
        (rigidbody3.get_n2(r3).x, rigidbody3.get_n2(r3).y),
        2.02,
        alpha=0.2,
        fc="yellow",
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
        r2[0],
        r2[1],
        c="k",
        alpha=1.0,
        s=120,
        edgecolor="w",
        zorder=4,
    )
    ax.scatter(
        r3[0],
        r3[1],
        c="k",
        alpha=1.0,
        s=120,
        edgecolor="w",
        zorder=4,
    )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.axis("off")
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_xticks([])
    ax.set_yticks([])

    fig.tight_layout()
    fig.savefig(
        f"{outname}",
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()


@dataclass
class Point:
    x: float
    y: float

    def as_array(self) -> np.ndarray:
        return np.array((self.x, self.y))


class RigidBody:
    """Define the rigid body for the optimisation."""

    def get_n1(self, r: np.ndarray, phi: float) -> Point:
        if phi != 0:
            # From origin x, y.
            x = self._n1[0]
            y = self._n1[1]
            # Rotated.
            axis = np.array((0, 0, 1))
            rot_mat = stk.rotation_matrix_arbitrary_axis(np.radians(phi), axis)

            rotated = rot_mat @ np.array((x, y, 0))

            # Add r.
            x, y = rotated[:2] + r

        else:
            x = self._n1[0] + r[0]
            y = self._n1[1] + r[1]
        return Point(x, y)

    def get_n2(self, r: np.ndarray, phi: float) -> Point:
        if phi != 0:
            # From origin x, y.
            x = self._n2[0]
            y = self._n2[1]
            # Rotated.
            axis = np.array((0, 0, 1))
            rot_mat = stk.rotation_matrix_arbitrary_axis(np.radians(phi), axis)

            rotated = rot_mat @ np.array((x, y, 0))

            # Add r.
            x, y = rotated[:2] + r
        else:
            x = self._n2[0] + r[0]
            y = self._n2[1] + r[1]
        return Point(x, y)

    def get_x1(self, r: np.ndarray, phi: float) -> Point:
        if phi != 0:
            # From origin x, y.
            x = self._n1[0] + self._x1[0]
            y = self._n1[1] + self._x1[1]
            # Rotated.
            axis = np.array((0, 0, 1))
            rot_mat = stk.rotation_matrix_arbitrary_axis(np.radians(phi), axis)

            rotated = rot_mat @ np.array((x, y, 0))

            # Add r.
            x, y = rotated[:2] + r
        else:
            x = self._n1[0] + r[0] + self._x1[0]
            y = self._n1[1] + r[1] + self._x1[1]
        return Point(x, y)

    def get_x2(self, r: np.ndarray, phi: float) -> Point:
        if phi != 0:
            # From origin x, y.
            x = self._n2[0] + self._x2[0]
            y = self._n2[1] + self._x2[1]
            # Rotated.
            axis = np.array((0, 0, 1))
            rot_mat = stk.rotation_matrix_arbitrary_axis(np.radians(phi), axis)

            rotated = rot_mat @ np.array((x, y, 0))

            # Add r.
            x, y = rotated[:2] + r
        else:
            x = self._n2[0] + r[0] + self._x2[0]
            y = self._n2[1] + r[1] + self._x2[1]
        return Point(x, y)


class LhsRigidBody(RigidBody):
    """Define the rigid body for the optimisation."""

    def __init__(
        self,
        nn_length: float,
        theta1: float,
        theta2: float,
        vector_length: float,
    ) -> None:
        self._vector_length = vector_length
        self._nn_length = nn_length
        self._n1 = np.array((0, -self._nn_length / 2))
        self._n2 = np.array((0, self._nn_length / 2))
        self._theta1 = theta1
        self._theta2 = theta2
        self._vector_length = vector_length
        if theta1 >= 90:
            self._x1 = np.array(
                (
                    vector_length * np.cos(np.radians(theta1 - 90)),
                    -vector_length * np.sin(np.radians(theta1 - 90)),
                )
            )
        else:
            self._x1 = np.array(
                (
                    vector_length * np.cos(np.radians(90 - theta1)),
                    +vector_length * np.sin(np.radians(90 - theta1)),
                )
            )

        if theta2 >= 90:
            self._x2 = np.array(
                (
                    vector_length * np.cos(np.radians(theta2 - 90)),
                    vector_length * np.sin(np.radians(theta2 - 90)),
                )
            )
        else:
            self._x2 = np.array(
                (
                    vector_length * np.cos(np.radians(90 - theta2)),
                    -vector_length * np.sin(np.radians(90 - theta2)),
                )
            )


class RhsRigidBody(RigidBody):
    """Define the rigid body for the optimisation."""

    def __init__(
        self,
        nn_length: float,
        theta1: float,
        theta2: float,
        vector_length: float,
    ) -> None:
        self._vector_length = vector_length
        self._nn_length = nn_length
        self._n1 = np.array((0, -self._nn_length / 2))
        self._n2 = np.array((0, self._nn_length / 2))
        self._theta1 = theta1
        self._theta2 = theta2
        self._vector_length = vector_length
        if theta1 >= 90:
            self._x1 = np.array(
                (
                    -vector_length * np.cos(np.radians(theta1 - 90)),
                    -vector_length * np.sin(np.radians(theta1 - 90)),
                )
            )
        else:
            self._x1 = np.array(
                (
                    -vector_length * np.cos(np.radians(90 - theta1)),
                    vector_length * np.sin(np.radians(90 - theta1)),
                )
            )

        if theta2 >= 90:
            self._x2 = np.array(
                (
                    -vector_length * np.cos(np.radians(theta2 - 90)),
                    vector_length * np.sin(np.radians(theta2 - 90)),
                )
            )
        else:
            self._x2 = np.array(
                (
                    -vector_length * np.cos(np.radians(90 - theta2)),
                    -vector_length * np.sin(np.radians(90 - theta2)),
                )
            )


@dataclass
class Pair:
    lhs: LhsRigidBody
    rhs: RhsRigidBody

    def calculate_residual(
        self,
        r1: float,
        phi1: float,
        r2: float,
        phi2: float,
    ) -> float:
        distance_residual = np.linalg.norm(
            self.rhs.get_x1(r2, phi2).as_array() - self.lhs.get_x1(r1, phi1).as_array()
        ) + np.linalg.norm(
            self.rhs.get_x2(r2, phi2).as_array() - self.lhs.get_x2(r1, phi1).as_array()
        )

        rhs_x1n1 = (
            self.rhs.get_x1(r2, phi2).as_array() - self.rhs.get_n1(r2, phi2).as_array()
        )
        rhs_x2n2 = (
            self.rhs.get_x2(r2, phi2).as_array() - self.rhs.get_n2(r2, phi2).as_array()
        )
        lhs_x1n1 = (
            self.lhs.get_x1(r1, phi1).as_array() - self.lhs.get_n1(r1, phi1).as_array()
        )
        lhs_x2n2 = (
            self.lhs.get_x2(r1, phi1).as_array() - self.lhs.get_n2(r1, phi1).as_array()
        )

        # They should be aligned, but opposite.
        angle_1_residual = abs(
            np.degrees(stko.vector_angle(vector1=lhs_x1n1, vector2=rhs_x1n1)) - 180
        )
        angle_2_residual = abs(
            np.degrees(stko.vector_angle(vector1=lhs_x2n2, vector2=rhs_x2n2)) - 180
        )

        angle_residual = angle_1_residual + angle_2_residual

        return distance_residual / 2 + angle_residual / 20


@dataclass
class PairResult:
    rigidbody1: LhsRigidBody
    rigidbody2: RhsRigidBody
    rigidbody3: RhsRigidBody
    state_1_result: float
    state_2_result: float
    state_1_parameters: abc.Sequence[float]
    state_2_parameters: abc.Sequence[float]
    set_parameters: abc.Sequence[float]
    rigidbody1 = LhsRigidBody(
        nn_length=c_dict1["NN_distance"],
        theta1=c_dict1["NN_BCN_angles"][0],
        theta2=c_dict1["NN_BCN_angles"][1],
        vector_length=vector_length(),
    )

    rigidbody2 = RhsRigidBody(
        nn_length=c_dict2["NN_distance"],
        theta1=c_dict2["NN_BCN_angles"][0],
        theta2=c_dict2["NN_BCN_angles"][1],
        vector_length=vector_length(),
    )

    rigidbody3 = RhsRigidBody(
        nn_length=c_dict2["NN_distance"],
        theta1=c_dict2["NN_BCN_angles"][1],
        theta2=c_dict2["NN_BCN_angles"][0],
        vector_length=vector_length(),
    )

    set_state = [-4, 0, 0]

    # Opt state 1.
    def f(params):
        r1x, r1y, phi1 = set_state
        r2x, r2y, phi2 = params
        return Pair(lhs=rigidbody1, rhs=rigidbody2).calculate_residual(
            r1=np.array((r1x, r1y)), phi1=phi1, r2=np.array((r2x, r2y)), phi2=phi2
        )

    # initial_guesses = ([-4, 0, 0, 0, 0, 0], [-4, 0, 0, 0, 0, 20], [-4, 0, 0, 0, 0, -20])
    initial_guesses = ([0, 0, 20], [0, 0, -20])  # , [0, 0, 0], )
    state_1_results = []
    state_1_sets = []
    for initial_guess in initial_guesses:
        result = optimize.minimize(
            f,
            initial_guess,
            # bounds=((-10, 0), (-5, 10), (-90, 90), (-5, 10), (-10, 0), (-90, 90)),
            bounds=((-5, 10), (-10, 0), (-45, 45)),
        )

        state_1_sets.append(result.x)
        state_1_results.append(
            Pair(lhs=rigidbody1, rhs=rigidbody2).calculate_residual(
                # r1=np.array((result.x[0], result.x[1])),
                # phi1=result.x[2],
                # r2=np.array((result.x[3], result.x[4])),
                # phi2=result.x[5],
                r1=np.array((set_state[0], set_state[1])),
                phi1=set_state[2],
                r2=np.array((result.x[0], result.x[1])),
                phi2=result.x[2],
            )
        )

    state_1_result = min(state_1_results)
    state_1_parameters = state_1_sets[state_1_results.index(state_1_result)]

    # Opt state 2.
    def f(params):
        r1x, r1y, phi1 = set_state
        r2x, r2y, phi2 = params
        return Pair(lhs=rigidbody1, rhs=rigidbody3).calculate_residual(
            r1=np.array((r1x, r1y)), phi1=phi1, r2=np.array((r2x, r2y)), phi2=phi2
        )

    state_2_results = []
    state_2_sets = []
    for initial_guess in initial_guesses:
        result = optimize.minimize(
            f,
            initial_guess,
            # bounds=((-10, 0), (-5, 10), (-90, 90), (-5, 10), (-10, 0), (-90, 90)),
            bounds=((-5, 10), (-10, 0), (-45, 45)),
        )

        state_2_sets.append(result.x)
        state_2_results.append(
            Pair(lhs=rigidbody1, rhs=rigidbody3).calculate_residual(
                # r1=np.array((result.x[0], result.x[1])),
                # phi1=result.x[2],
                # r2=np.array((result.x[3], result.x[4])),
                # phi2=result.x[5],
                r1=np.array((set_state[0], set_state[1])),
                phi1=set_state[2],
                r2=np.array((result.x[0], result.x[1])),
                phi2=result.x[2],
            )
        )

    state_2_result = min(state_2_results)
    state_2_parameters = state_2_sets[state_2_results.index(state_2_result)]

    # plot_pair_position(
    #     # r1=np.array((state_1_parameters[0], state_1_parameters[1])),
    #     # phi1=state_1_parameters[2],
    #     # rigidbody1=rigidbody1,
    #     # r2=np.array((state_1_parameters[3], state_1_parameters[4])),
    #     # phi2=state_1_parameters[5],
    #     # rigidbody2=rigidbody2,
    #     # r3=np.array((state_2_parameters[3], state_2_parameters[4])),
    #     # phi3=state_2_parameters[5],
    #     r1=np.array((set_state[0], set_state[1])),
    #     phi1=set_state[2],
    #     rigidbody1=rigidbody1,
    #     r2=np.array((state_1_parameters[0], state_1_parameters[1])),
    #     phi2=state_1_parameters[2],
    #     rigidbody2=rigidbody2,
    #     r3=np.array((state_2_parameters[0], state_2_parameters[1])),
    #     phi3=state_2_parameters[2],
    #     rigidbody3=rigidbody3,
    #     outname="test.png",
    # )

    return PairResult(
        rigidbody1=rigidbody1,
        rigidbody2=rigidbody2,
        rigidbody3=rigidbody3,
        state_1_result=state_1_result,
        state_2_result=state_2_result,
        state_1_parameters=state_1_parameters,
        state_2_parameters=state_2_parameters,
        set_parameters=set_state,
    )
