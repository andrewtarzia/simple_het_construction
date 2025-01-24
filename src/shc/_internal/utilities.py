"""Module for utility functions."""

from collections import abc

import atomlite
import numpy as np
import stk
from rdkit.Chem import AllChem as rdkit  # noqa: N813  # noqa: N813
from rdkit.Chem import rdMolTransforms

from shc.geometry import PairResult


def merge_stk_molecules(
    molecules: abc.Sequence[stk.Molecule],
) -> stk.BuildingBlock:
    """Merge any list of separate molecules to be in one class.

    TODO: add to stko.

    """
    atoms = []
    bonds = []
    pos_mat = []
    for molecule in molecules:
        atom_ids_map = {}
        for atom in molecule.get_atoms():
            new_id = len(atoms)
            atom_ids_map[atom.get_id()] = new_id
            atoms.append(
                stk.Atom(
                    id=atom_ids_map[atom.get_id()],
                    atomic_number=atom.get_atomic_number(),
                    charge=atom.get_charge(),
                )
            )

        bonds.extend(
            i.with_ids(id_map=atom_ids_map) for i in molecule.get_bonds()
        )
        pos_mat.extend(list(molecule.get_position_matrix()))

    return stk.BuildingBlock.init(
        atoms=atoms,
        bonds=bonds,
        position_matrix=np.array(pos_mat),
    )


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

    TODO: Add to stko.

    """
    pos_mat = rdk_mol.GetConformer(id=conf_id).GetPositions()
    return stk_mol.with_position_matrix(pos_mat)


def extract_torsions(
    molecule: stk.Molecule,
    smarts: str,
    expected_num_atoms: int,
    scanned_ids: tuple[int, int, int, int],
    expected_num_torsions: int,
) -> tuple[float, float]:
    """Extract two torsions from a molecule."""
    rdkit_molecule = molecule.to_rdkit_mol()
    matches = rdkit_molecule.GetSubstructMatches(
        query=rdkit.MolFromSmarts(smarts),
    )

    torsions = []
    for match in matches:
        if len(match) != expected_num_atoms:
            msg = f"{len(match)} not as expected ({expected_num_atoms})"
            raise RuntimeError(msg)
        torsions.append(
            abs(
                rdMolTransforms.GetDihedralDeg(
                    rdkit_molecule.GetConformer(0),
                    match[scanned_ids[0]],
                    match[scanned_ids[1]],
                    match[scanned_ids[2]],
                    match[scanned_ids[3]],
                )
            )
        )
    if len(torsions) != expected_num_torsions:
        msg = f"{len(torsions)} found, not {expected_num_torsions}!"
        raise RuntimeError(msg)
    return tuple(torsions)


def get_amide_torsions(molecule: stk.Molecule) -> tuple[float, float]:
    """Get the centre alkene torsion from COOH."""
    smarts = "[#6X3H0]~[#6X3H1]~[#6X3H0]~[#6X3H0](=[#8])~[#7]"
    expected_num_atoms = 6
    scanned_ids = (1, 2, 3, 5)

    return extract_torsions(
        molecule=molecule,
        smarts=smarts,
        expected_num_atoms=expected_num_atoms,
        scanned_ids=scanned_ids,
        expected_num_torsions=1,
    )


def get_num_alkynes(rdkit_mol: rdkit.Mol) -> int:
    """Get the number of alkynes in a molecule."""
    smarts = "[#6]#[#6]"
    matches = rdkit_mol.GetSubstructMatches(
        query=rdkit.MolFromSmarts(smarts),
    )
    return len(matches)


def remake_atomlite_molecule(molecule: atomlite.Molecule) -> atomlite.Molecule:
    """Update to newest scheme."""
    new_molecule = molecule.copy()
    new_molecule["atomic_numbers"] = [7, 7, 1, 1, 7, 7, 1, 1]

    return new_molecule


def to_atomlite_molecule(
    pair: PairResult,
    best_state: int,
) -> atomlite.Molecule:
    """Convert a pair result to an atomlite molecule."""
    if best_state == 1:
        r1 = np.array((pair.set_parameters[0], pair.set_parameters[1]))
        phi1 = pair.set_parameters[2]
        rigidbody1 = pair.rigidbody1
        r2 = np.array((pair.state_1_parameters[0], pair.state_1_parameters[1]))
        phi2 = pair.state_1_parameters[2]
        rigidbody2 = pair.rigidbody2
    elif best_state == 2:  # noqa: PLR2004
        r1 = np.array((pair.set_parameters[0], pair.set_parameters[1]))
        phi1 = pair.set_parameters[2]
        rigidbody1 = pair.rigidbody1
        r2 = np.array((pair.state_2_parameters[0], pair.state_2_parameters[1]))
        phi2 = pair.state_2_parameters[2]
        rigidbody2 = pair.rigidbody3

    l_n1 = [*rigidbody1.get_n1(r1, phi1).as_list(), 0]
    l_n2 = [*rigidbody1.get_n2(r1, phi1).as_list(), 0]
    l_x1 = [*rigidbody1.get_x1(r1, phi1).as_list(), 0]
    l_x2 = [*rigidbody1.get_x2(r1, phi1).as_list(), 0]
    r_n1 = [*rigidbody2.get_n1(r2, phi2).as_list(), 0]
    r_n2 = [*rigidbody2.get_n2(r2, phi2).as_list(), 0]
    r_x1 = [*rigidbody2.get_x1(r2, phi2).as_list(), 0]
    r_x2 = [*rigidbody2.get_x2(r2, phi2).as_list(), 0]

    atomic_numbers = [7, 7, 1, 1, 7, 7, 1, 1]
    atom_charges = [0] * 8
    bonds = atomlite.Bonds = {
        "atom1": [0, 0, 1, 4, 4, 5],
        "atom2": [1, 2, 3, 5, 6, 7],
        "order": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    }
    molecule: atomlite.Molecule = {
        "atomic_numbers": atomic_numbers,
        "conformers": [[l_n1, l_n2, l_x1, l_x2, r_n1, r_n2, r_x1, r_x2]],
        "atom_charges": atom_charges,
        "bonds": bonds,
    }

    return molecule
