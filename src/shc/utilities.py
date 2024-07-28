"""Module for utility functions."""

import numpy as np
import stk
import stko
from rdkit.Chem import AllChem as rdkit  # noqa: N813


def update_from_rdkit_conf(
    stk_mol: stk.Molecule,
    rdk_mol: rdkit.Mol,
    conf_id: int,
) -> stk.Molecule:
    """Update the structure to match `conf_id` of `mol`.

    Parameters:
        struct:
            The molecule whoce coordinates are to be updated.

        mol:
            The :mod:`rdkit` molecule to use for the structure update.

        conf_id:
            The conformer ID of the `mol` to update from.

    Returns:
        The molecule.

    """
    pos_mat = rdk_mol.GetConformer(id=conf_id).GetPositions()
    return stk_mol.with_position_matrix(pos_mat)


class AromaticCNCFactory(stk.FunctionalGroupFactory):
    """A subclass of stk.SmartsFunctionalGroupFactory."""

    def __init__(
        self,
        bonders: tuple[int, ...] = (1,),
        deleters: tuple[int, ...] = (),
    ) -> None:
        """Initialise :class:`.AromaticCNCFactory`."""
        self._bonders = bonders
        self._deleters = deleters

    def get_functional_groups(self, molecule: stk.Molecule):  # noqa: ANN201
        """Get functional groups."""
        generic_functional_groups = stk.SmartsFunctionalGroupFactory(
            smarts="[#6]~[#7X2]~[#6]",
            bonders=self._bonders,
            deleters=self._deleters,
        ).get_functional_groups(molecule)
        for fg in generic_functional_groups:
            atom_ids = (i.get_id() for i in fg.get_atoms())
            atoms = tuple(molecule.get_atoms(atom_ids))
            yield AromaticCNC(
                carbon1=atoms[0],
                nitrogen=atoms[1],
                carbon2=atoms[2],
                bonders=tuple(atoms[i] for i in self._bonders),
                deleters=tuple(atoms[i] for i in self._deleters),
            )


class AromaticCNC(stk.GenericFunctionalGroup):
    """Represents an N atom in pyridine functional group.

    The structure of the functional group is given by the pseudo-SMILES
    ``[carbon][nitrogen][carbon]``.

    """

    def __init__(
        self,
        carbon1,  # noqa: ANN001
        nitrogen,  # noqa: ANN001
        carbon2,  # noqa: ANN001
        bonders,  # noqa: ANN001
        deleters,  # noqa: ANN001
    ) -> None:
        """Initialize a :class:`.AromaticCNC` instance.

        Parameters
        ----------
        carbon1 : :class:`.C`
            The first carbon atom.

        nitrogen : :class:`.N`
            The nitrogen atom.

        carbon2 : :class:`.C`
            The second carbon atom.

        bonders : :class:`tuple` of :class:`.Atom`
            The bonder atoms.

        deleters : :class:`tuple` of :class:`.Atom`
            The deleter atoms.

        """
        self._carbon1 = carbon1
        self._nitrogen = nitrogen
        self._carbon2 = carbon2
        atoms = (carbon1, nitrogen, carbon2)
        super().__init__(atoms, bonders, deleters)

    def get_carbon1(self) -> stk.Atom:
        """Get atom."""
        return self._carbon1

    def get_carbon2(self) -> stk.Atom:
        """Get atom."""
        return self._carbon2

    def get_nitrogen(self) -> stk.Atom:
        """Get atom."""
        return self._nitrogen

    def clone(self) -> stk.GenericFunctionalGroup:
        """Clone functional group."""
        clone = super().clone()
        clone._carbon1 = self._carbon1  # noqa: SLF001
        clone._nitrogen = self._nitrogen  # noqa: SLF001
        clone._carbon2 = self._carbon2  # noqa: SLF001
        return clone

    def with_atoms(self, atom_map) -> stk.GenericFunctionalGroup:  # noqa: ANN001
        """Change atoms."""
        clone = super().with_atoms(atom_map)
        clone._carbon1 = atom_map.get(  # noqa: SLF001
            self._carbon1.get_id(),
            self._carbon1,
        )
        clone._nitrogen = atom_map.get(  # noqa: SLF001
            self._nitrogen.get_id(),
            self._nitrogen,
        )
        clone._carbon2 = atom_map.get(  # noqa: SLF001
            self._carbon2.get_id(),
            self._carbon2,
        )
        return clone

    def __repr__(self) -> str:
        """Get repr."""
        return (
            f"{self.__class__.__name__}("
            f"{self._carbon1}, {self._nitrogen}, {self._carbon2}, "
            f"bonders={self._bonders})"
        )


def calculate_n_centroid_n_angle(bb: stk.BuildingBlock) -> float:
    """Calculate the N-centroid-N angle of a ditopic building block.

    This function will not work for cages built from FGs other than
    metals + AromaticCNC and metals + AromaticCNN.

    Warning:
        This is an old version of this function. Use the stko function instead,
        line in study 2. But caution that there are some changes in
        definitions.

    Parameters:
        bb:
            stk molecule to analyse.

    Returns:
        angle:
            Angle between two bonding vectors of molecule.

    """
    fg_counts = 0
    fg_positions = []
    for fg in bb.get_functional_groups():
        if isinstance(fg, AromaticCNC):
            fg_counts += 1
            # Get geometrical properties of the FG.
            # Get N position - deleter.
            (n_position,) = bb.get_atomic_positions(
                atom_ids=fg.get_nitrogen().get_id()
            )
            fg_positions.append(n_position)

    if fg_counts != 2:  # noqa: PLR2004
        msg = f"{bb} does not have 2 ThreesiteFG functional groups."
        raise ValueError(msg)

    # Get building block centroid.
    centroid_position = bb.get_centroid()

    # Get vectors.
    fg_vectors = [i - centroid_position for i in fg_positions]

    # Calculate the angle between the two vectors.
    return np.degrees(stko.vector_angle(*fg_vectors))


def calculate_nn_distance(bb: stk.BuildingBlock) -> float:
    """Calculate the N-N distance of ditopic building block.

    This function will not work for cages built from FGs other than
    metals + AromaticCNC.

    Warning:
        This is an old version of this function. Use the stko function instead,
        line in study 2. But caution that there are some changes in
        definitions.

    Parameters:
        bb:
            stk molecule to analyse.

    Returns:
        Distance(s) between [angstrom] N atoms in functional groups.

    """
    fg_counts = 0
    n_positions = []
    for fg in bb.get_functional_groups():
        if isinstance(fg, AromaticCNC):
            fg_counts += 1
            # Get geometrical properties of the FG.
            # Get N position - deleter.
            (n_position,) = bb.get_atomic_positions(
                atom_ids=fg.get_nitrogen().get_id()
            )
            n_positions.append(n_position)

    if fg_counts != 2:  # noqa: PLR2004
        msg = f"{bb} does not have 2 ThreesiteFG functional groups."
        raise ValueError(msg)

    return np.linalg.norm(n_positions[0] - n_positions[1])


def calculate_nn_bcn_angles(bb: stk.BuildingBlock) -> dict[str, float]:
    """Calculate binder-NN angles.

    Warning:
        This is an old version of this function. Use the stko function instead,
        line in study 2. But caution that there are some changes in
        definitions. The vector direction definitions change.
    """
    fg_counts = 0
    n_positions = []
    n_c_vectors = []
    for fg in bb.get_functional_groups():
        if isinstance(fg, AromaticCNC):
            fg_counts += 1
            # Get geometrical properties of the FG.
            # Get N position - deleter.
            (n_position,) = bb.get_atomic_positions(
                atom_ids=fg.get_nitrogen().get_id()
            )
            n_positions.append(n_position)
            c_atom_ids = (
                fg.get_carbon1().get_id(),
                fg.get_carbon2().get_id(),
            )
            c_centroid = bb.get_centroid(atom_ids=c_atom_ids)
            n_c_vector = n_position - c_centroid
            n_c_vectors.append(n_c_vector)

    if fg_counts != 2:  # noqa: PLR2004
        msg = f"{bb} does not have 2 ThreesiteFG functional groups."
        raise ValueError(msg)

    nn_vector = n_positions[1] - n_positions[0]

    nn_bcn_1 = np.degrees(stko.vector_angle(n_c_vectors[0], -nn_vector))
    nn_bcn_2 = np.degrees(stko.vector_angle(n_c_vectors[1], nn_vector))

    return {"NN_BCN1": nn_bcn_1, "NN_BCN2": nn_bcn_2}


def calculate_nccn_dihedral(bb: stk.BuildingBlock) -> float:
    """Calculate NCCN dihedral.

    Warning:
        This is an old version of this function. Use the stko function instead,
        line in study 2. But caution that there are some changes in
        definitions.

    """
    fg_counts = 0
    n_positions = []
    c_centroids = []
    for fg in bb.get_functional_groups():
        if isinstance(fg, AromaticCNC):
            fg_counts += 1
            # Get geometrical properties of the FG.
            # Get N position - deleter.
            (n_position,) = bb.get_atomic_positions(
                atom_ids=fg.get_nitrogen().get_id()
            )
            n_positions.append(n_position)
            c_atom_ids = (
                fg.get_carbon1().get_id(),
                fg.get_carbon2().get_id(),
            )
            c_centroid = bb.get_centroid(atom_ids=c_atom_ids)
            c_centroids.append(c_centroid)

    if fg_counts != 2:  # noqa: PLR2004
        msg = f"{bb} does not have 2 ThreesiteFG functional groups."
        raise ValueError(msg)

    return stko.calculate_dihedral(
        pt1=n_positions[0],
        pt2=c_centroids[0],
        pt3=c_centroids[1],
        pt4=n_positions[1],
    )
