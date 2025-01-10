"""Module for geometry functions."""

from collections import abc
from dataclasses import dataclass

import numpy as np
import stk
import stko


@dataclass
class Point:
    """Define a point in rigid body."""

    x: float
    y: float

    def as_array(self) -> np.ndarray:
        """Get point as array."""
        return np.array((self.x, self.y))

    def as_list(self) -> list:
        """Get point as list."""
        return [self.x, self.y]


class RigidBody:
    """Define the rigid body for the optimisation."""

    def get_n1(self, r: np.ndarray, phi: float) -> Point:
        """Get n1 point."""
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
        """Get n2 point."""
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
        """Get x1 point."""
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
        """Get x2 point."""
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
        """Initialise rigid body."""
        self._vector_length = vector_length
        self._nn_length = nn_length
        self._n1 = np.array((0, -self._nn_length / 2))
        self._n2 = np.array((0, self._nn_length / 2))
        self._theta1 = theta1
        self._theta2 = theta2
        self._vector_length = vector_length
        if theta1 >= 90:  # noqa: PLR2004
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

        if theta2 >= 90:  # noqa: PLR2004
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
        """Initialise rigid body."""
        self._vector_length = vector_length
        self._nn_length = nn_length
        self._n1 = np.array((0, -self._nn_length / 2))
        self._n2 = np.array((0, self._nn_length / 2))
        self._theta1 = theta1
        self._theta2 = theta2
        self._vector_length = vector_length
        if theta1 >= 90:  # noqa: PLR2004
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

        if theta2 >= 90:  # noqa: PLR2004
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
    """Pair of rigid bodies."""

    lhs: LhsRigidBody
    rhs: RhsRigidBody

    def calculate_residual(  # noqa: PLR0913
        self,
        r1: float,
        phi1: float,
        r2: float,
        phi2: float,
        k_bond: float,
        k_angle: float,
    ) -> float:
        """Calculate the residual."""
        distance_1_residual = np.linalg.norm(
            self.rhs.get_x1(r2, phi2).as_array()
            - self.lhs.get_x1(r1, phi1).as_array()
        )
        distance_2_residual = np.linalg.norm(
            self.rhs.get_x2(r2, phi2).as_array()
            - self.lhs.get_x2(r1, phi1).as_array()
        )
        # Distance in angstrom. Target is 0.
        distance_term = ((0.5) * k_bond * distance_1_residual**2) + (
            (0.5) * k_bond * distance_2_residual**2
        )

        rhs_x1n1 = (
            self.rhs.get_x1(r2, phi2).as_array()
            - self.rhs.get_n1(r2, phi2).as_array()
        )
        rhs_x2n2 = (
            self.rhs.get_x2(r2, phi2).as_array()
            - self.rhs.get_n2(r2, phi2).as_array()
        )
        lhs_x1n1 = (
            self.lhs.get_x1(r1, phi1).as_array()
            - self.lhs.get_n1(r1, phi1).as_array()
        )
        lhs_x2n2 = (
            self.lhs.get_x2(r1, phi1).as_array()
            - self.lhs.get_n2(r1, phi1).as_array()
        )

        # They should be aligned, but opposite.
        # Residuals from 180 in degrees.
        angle_1_residual = (
            np.degrees(stko.vector_angle(vector1=lhs_x1n1, vector2=rhs_x1n1))
            - 180
        )
        angle_2_residual = (
            np.degrees(stko.vector_angle(vector1=lhs_x2n2, vector2=rhs_x2n2))
            - 180
        )

        # Harmonic angle term.
        angle_1_term = (0.5) * k_angle * (np.radians(angle_1_residual)) ** 2
        angle_2_term = (0.5) * k_angle * (np.radians(angle_2_residual)) ** 2
        angle_residual = angle_1_term + angle_2_term

        return distance_term + angle_residual


@dataclass
class PairResult:
    """Result of pair comparison."""

    rigidbody1: LhsRigidBody
    rigidbody2: RhsRigidBody
    rigidbody3: RhsRigidBody
    state_1_result: float
    state_2_result: float
    state_1_parameters: abc.Sequence[float]
    state_2_parameters: abc.Sequence[float]
    set_parameters: abc.Sequence[float]
