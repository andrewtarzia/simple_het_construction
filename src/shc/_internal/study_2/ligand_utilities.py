"""Script to build the ligand in this project."""

import logging
import pathlib
import time
from collections import abc

import atomlite
import bbprep
import numpy as np
import stk
import stko
from rdkit.Chem import AllChem as rdkit  # noqa: N813
from rdkit.Chem import Draw
from rmsd import kabsch_rmsd

from shc.definitions import EnvVariables
from shc.utilities import update_from_rdkit_conf


def symmetry_check(
    building_blocks: abc.Sequence[stk.BuildingBlock],
    composition: str,
    repeating_unit: str,
) -> bool:
    """Check if a ligand will be symmetric."""
    base = ord("A")
    ru = tuple(ord(letter) - base for letter in repeating_unit)
    bb_smiles = [stk.Smiles().get_key(i) for i in building_blocks]

    if composition == "ae":
        return bb_smiles[0] == bb_smiles[1]

    if composition in ("ace", "abe", "ace", "ade"):
        return bb_smiles[ru[0]] == bb_smiles[ru[2]]

    if composition in ("abce", "adce", "acbe", "acde"):
        outers = bb_smiles[ru[0]] == bb_smiles[ru[3]]
        inners = bb_smiles[ru[1]] == bb_smiles[ru[2]]
        return outers and inners

    if composition in ("abca", "ebce", "adca", "edce"):
        return bb_smiles[ru[1]] == bb_smiles[ru[2]]

    if composition in ("abcde", "adcbe", "abcbe", "adcde"):
        outers = bb_smiles[ru[0]] == bb_smiles[ru[4]]
        inners = bb_smiles[ru[1]] == bb_smiles[ru[3]]
        return outers and inners

    if composition in ("abcda", "ebcde"):
        return bb_smiles[ru[1]] == bb_smiles[ru[3]]

    msg = f"missing definition for {composition}"
    raise NotImplementedError(msg)


def normalise_names(name: str) -> str:
    """Normalise names.

    No normalisation actually needed based on iteration.
    """
    # Flip order of building blocks if binder 1 is > id than binder 2.
    bbs = name.split("_")[1].split("-")

    if bbs[0] == "x" or bbs[-1] == "x":
        return name

    return name


def passes_dedupe(molecule: stk.Molecule, key_db: atomlite.Database) -> bool:
    """Check if ligand is in dedupe databases."""
    smiles = stk.Smiles().get_key(molecule)
    if key_db.has_entry(smiles):
        return False

    inchi = stk.Inchi().get_key(molecule)
    if key_db.has_entry(inchi):
        return False

    inchi_key = stk.InchiKey().get_key(molecule)
    if key_db.has_entry(inchi_key):  # noqa: SIM103
        return False

    return True


def update_keys_db(molecule: stk.Molecule, key_db: atomlite.Database) -> None:
    """Update keys db."""
    smiles = stk.Smiles().get_key(molecule)
    inchi = stk.Inchi().get_key(molecule)
    inchi_key = stk.InchiKey().get_key(molecule)
    key_db.add_entries(
        entries=(
            atomlite.Entry(
                key=smiles,
                molecule=atomlite.json_from_rdkit(molecule.to_rdkit_mol()),
            ),
            atomlite.Entry(
                key=inchi,
                molecule=atomlite.json_from_rdkit(molecule.to_rdkit_mol()),
            ),
            atomlite.Entry(
                key=inchi_key,
                molecule=atomlite.json_from_rdkit(molecule.to_rdkit_mol()),
            ),
        )
    )


def build_ligand(  # noqa: PLR0913
    building_blocks: abc.Sequence[stk.BuildingBlock],
    repeating_unit: str,
    ligand_name: str,
    ligand_dir: pathlib.Path,
    figures_dir: pathlib.Path,
    deduped_db: atomlite.Database,
    key_db: atomlite.Database,
) -> None:
    """Build a ligand."""
    lowe_file = ligand_dir / f"{ligand_name}_lowe.mol"
    if lowe_file.exists():
        molecule = stk.BuildingBlock.init_from_file(lowe_file)
    else:
        molecule = stk.ConstructedMolecule(
            topology_graph=stk.polymer.Linear(
                building_blocks=building_blocks,
                repeating_unit=repeating_unit,
                num_repeating_units=1,
                num_processes=1,
            )
        )

    if passes_dedupe(molecule=molecule, key_db=key_db):
        Draw.MolToFile(
            rdkit.MolFromSmiles(stk.Smiles().get_key(molecule)),
            figures_dir / f"{ligand_name}_2d_new.png",
            size=(300, 300),
        )
        molecule = stko.ETKDG().optimize(molecule)

        # Check if in db.
        if not deduped_db.has_entry(key=ligand_name):
            explore_ligand(
                molecule=molecule,
                ligand_name=ligand_name,
                ligand_dir=ligand_dir,
                ligand_db=deduped_db,
            )

        update_keys_db(molecule, key_db)


def explore_ligand(
    molecule: stk.Molecule,
    ligand_name: str,
    ligand_dir: pathlib.Path,
    ligand_db: atomlite.Database,
) -> None:
    """Do conformer scan."""
    st = time.time()
    conf_database_path = ligand_dir / f"confs_{ligand_name}.db"

    logging.info("building conformer ensemble of %s", ligand_name)
    confs = molecule.to_rdkit_mol()
    etkdg = rdkit.srETKDGv3()
    etkdg.randomSeed = 1000
    cids = rdkit.EmbedMultipleConfs(mol=confs, numConfs=500, params=etkdg)

    lig_conf_data = {}
    num_confs_kept = 0
    conformers_kept = []
    min_energy = float("inf")
    for cid in cids:
        conf_key = f"{ligand_name}_{cid}_aa"

        # Update stk_mol to conformer geometry.
        new_mol = update_from_rdkit_conf(
            stk_mol=molecule,
            rdk_mol=confs,
            conf_id=cid,
        )
        # Need to define the functional groups.
        new_mol = stk.BuildingBlock.init_from_molecule(
            molecule=new_mol,
            functional_groups=stko.functional_groups.ThreeSiteFactory(
                smarts="[#6]~[#7X2]~[#6]", bonders=(1,), deleters=()
            ),
        )
        # Only get two FGs.
        new_mol = bbprep.FurthestFGs().modify(
            building_block=new_mol,
            desired_functional_groups=2,
        )

        new_mol = stko.UFF().optimize(mol=new_mol)
        energy = stko.UFFEnergy().get_energy(new_mol)
        if energy < min_energy:
            min_energy = energy
            new_mol.write(ligand_dir / f"{ligand_name}_lowe.mol")

        min_rmsd = float("inf")
        if len(conformers_kept) == 0:
            conformers_kept.append((cid, new_mol))

        else:
            # Get heavy-atom RMSD to all other conformers and check if it is
            # within threshold to any of them.
            for _, conformer in conformers_kept:
                rmsd = kabsch_rmsd(
                    np.array(
                        tuple(
                            conformer.get_atomic_positions(
                                atom_ids=tuple(
                                    i.get_id()
                                    for i in conformer.get_atoms()
                                    if i.get_atomic_number() != 1
                                ),
                            )
                        )
                    ),
                    np.array(
                        tuple(
                            new_mol.get_atomic_positions(
                                atom_ids=tuple(
                                    i.get_id()
                                    for i in new_mol.get_atoms()
                                    if i.get_atomic_number() != 1
                                ),
                            )
                        )
                    ),
                    translate=True,
                )

                min_rmsd = min((min_rmsd, rmsd))
                if min_rmsd < EnvVariables.rmsd_threshold:
                    break

        # If any RMSD is less than threshold, skip.
        if min_rmsd < EnvVariables.rmsd_threshold:
            continue

        conformers_kept.append((cid, new_mol))
        analyser = stko.molecule_analysis.DitopicThreeSiteAnalyser()
        lig_conf_data[cid] = {
            "NcentroidN_angle": analyser.get_binder_centroid_angle(new_mol),
            "NCCN_dihedral": analyser.get_binder_adjacent_torsion(new_mol),
            "NN_distance": analyser.get_binder_distance(new_mol),
            "NN_BCN_angles": analyser.get_binder_angles(new_mol),
            "UFFEnergy;kj/mol": energy * 4.184,
        }
        num_confs_kept += 1

        conf_entry = atomlite.Entry.from_rdkit(
            key=conf_key, molecule=new_mol.to_rdkit_mol()
        )

        atomlite.Database(conf_database_path).add_entries(conf_entry)

    try:
        lpattern = ligand_name.split("_")[0]
        cpattern = ligand_name.split("_")[1]
    except IndexError:
        lpattern = ligand_name
        cpattern = "not-built"

    entry = atomlite.Entry.from_rdkit(
        key=ligand_name,
        molecule=stk.BuildingBlock.init_from_file(
            ligand_dir / f"{ligand_name}_lowe.mol"
        ).to_rdkit_mol(),
        properties={
            "conf_data": lig_conf_data,
            "min_energy;kj/mol": min_energy * 4.184,
            "ligand_pattern": lpattern,
            "composition_pattern": cpattern,
        },
    )
    ligand_db.add_entries(entry)

    logging.info(
        "%s confs generated for %s, %s kept, in %s s",
        cid,
        ligand_name,
        num_confs_kept,
        round(time.time() - st, 2),
    )
