"""Script to build the ligand in this project."""

import logging

import pathlib

import stk
import itertools as it
import matplotlib.pyplot as plt
import atomlite
from definitions import EnvVariables
from study_2_build_ligands import explore_ligand
import stko

import numpy as np
from rdkit.Chem import AllChem as rdkit  # noqa: N813
from rdkit.Chem import rdMolTransforms, rdMolDescriptors, rdmolops
import argparse
import time
from matching_functions import angle_test, mismatch_test


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--plot_ligands",
        action="store_true",
        help="plot ligand data",
    )

    return parser.parse_args()


def analyse_ligand_pair(
    ligand1: str,
    ligand2: str,
    key: str,
    ligand_db: atomlite.Database,
    pair_db: atomlite.Database,
):
    ligand1_entry = ligand_db.get_entry(ligand1)
    ligand1_confs = ligand1_entry.properties["conf_data"]
    ligand2_entry = ligand_db.get_entry(ligand2)
    ligand2_confs = ligand2_entry.properties["conf_data"]
    st = time.time()
    num_pairs = 0
    min_geom_score = 1e24
    print("check this to pair count", len(ligand1_confs), len(ligand2_confs))
    pair_data = {}

    # Iterate over the product of all conformers.
    for cid_1, cid_2 in it.product(ligand1_confs, ligand2_confs):
        cid_name = f"{cid_1}-{cid_2}"

        # Check strain.
        logging.info("remove conversion")
        strain1 = (
            ligand1_confs[cid_1]["UFFEnergy;kj/mol"]
            - ligand1_entry.properties["min_energy;kj/mol"] * 4.184
        )
        strain2 = (
            ligand2_confs[cid_2]["UFFEnergy;kj/mol"]
            - ligand2_entry.properties["min_energy;kj/mol"] * 4.184
        )
        if strain1 > EnvVariables.strain_cutoff or strain2 > EnvVariables.strain_cutoff:
            continue

        # Check torsion.
        torsion1 = abs(ligand1_confs[cid_1]["NCCN_dihedral"])
        torsion2 = abs(ligand2_confs[cid_2]["NCCN_dihedral"])
        if (
            torsion1 > EnvVariables.dihedral_cutoff
            or torsion2 > EnvVariables.dihedral_cutoff
        ):
            continue

        # Calculate geom score for both sides together.
        c_dict1 = ligand1_confs[cid_1]
        c_dict2 = ligand2_confs[cid_2]
        print(c_dict1, c_dict2)
        print(cid_1, cid_2, key)

        logging.info(
            "should I pick large and small, or just define the alg so it does not matter?"
        )

        # Calculate final geometrical properties.
        # T1.
        angle_dev = angle_test(c_dict1=c_dict1, c_dict2=c_dict2)

        pair_results = mismatch_test(c_dict1=c_dict1, c_dict2=c_dict2)

        pair_data[cid_name] = {
            "state_1_residual": float(pair_results.state_1_result),
            "state_2_residual": float(pair_results.state_2_result),
            "state_1_parameters": [float(i) for i in pair_results.state_1_parameters],
            "state_2_parameters": [float(i) for i in pair_results.state_2_parameters],
            "ligand1_key": ligand1,
            "ligand2_key": ligand2,
            "cid_1": cid_1,
            "cid_2": cid_2,
            "torsion1": torsion1,
            "torsion2": torsion2,
            "strain1": strain1,
            "strain2": strain2,
            "angle_deviation": angle_dev,
        }

        pair_db.set_property(
            key=key,
            path=f"$.pair_data.{cid_name}.state_1_residual",
            property=float(pair_results.state_1_result),
            commit=False,
        )
        pair_db.set_property(
            key=key,
            path=f"$.pair_data.{cid_name}.state_2_residual",
            property=float(pair_results.state_2_result),
            commit=False,
        )
        pair_db.set_property(
            key=key,
            path=f"$.pair_data.{cid_name}.angle_deviation",
            property=angle_dev,
            commit=False,
        )
    ft = time.time()
    logging.info(
        f"time taken for pairing {ligand1}, {ligand2}: "
        f"{round(1000*(ft-st), 2)}ms "
        f"({round(1000*(ft-st)/num_pairs)}ms"
        f" per pair) - {num_pairs} pairs"
    )
    print(
        "check this to pair count",
        len(ligand1_confs),
        len(ligand2_confs),
        len(ligand1_confs) * len(ligand2_confs),
        num_pairs,
    )


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
    smarts = "[#6]#[#6]"
    matches = rdkit_mol.GetSubstructMatches(
        query=rdkit.MolFromSmarts(smarts),
    )
    return len(matches)


def plot_ligand(
    ligand_name: str,
    ligand_db: atomlite.Database,
    ligand_dir: pathlib.Path,
    figures_dir: pathlib.Path,
) -> None:
    nnmaps = {
        "lab_0": (9.8, 12.8),
        "la_0": (8.7,),
        "lb_0": (11,),
        "lc_0": (8.1,),
        "ld_0": (9.9,),
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    entry = ligand_db.get_entry(ligand_name)
    conf_data = entry.properties["conf_data"]
    logging.info("remove conversion")
    min_energy = entry.properties["min_energy;kj/mol"] * 4.184

    low_energy_states = [
        i
        for i in conf_data
        if (conf_data[i]["UFFEnergy;kj/mol"] - min_energy) < EnvVariables.strain_cutoff
    ]
    ax.scatter(
        [conf_data[i]["NN_distance"] for i in conf_data],
        [sum(conf_data[i]["NN_BCN_angles"]) for i in conf_data],
        c="tab:gray",
        s=20,
        ec="none",
        alpha=0.2,
        label="all",
    )

    if ligand_name == "lab_0":
        states = {"b": [], "f": []}
        conf_dir = ligand_dir / "confs_lab_0"
        for conf in low_energy_states:
            conf_mol = stk.BuildingBlock.init_from_file(
                conf_dir / f"lab_0_c{conf}_cuff.mol"
            )

            torsion_state = "f" if get_amide_torsions(conf_mol)[0] < 90 else "b"  # noqa: PLR2004

            states[torsion_state].append(conf)

        ax.scatter(
            [conf_data[i]["NN_distance"] for i in states["b"]],
            [sum(conf_data[i]["NN_BCN_angles"]) for i in states["b"]],
            c="tab:blue",
            s=20,
            ec="none",
            label="low energy, b",
        )
        ax.scatter(
            [conf_data[i]["NN_distance"] for i in states["f"]],
            [sum(conf_data[i]["NN_BCN_angles"]) for i in states["f"]],
            c="tab:orange",
            s=20,
            ec="none",
            label="low energy, f",
        )

    else:
        ax.scatter(
            [conf_data[i]["NN_distance"] for i in low_energy_states],
            [sum(conf_data[i]["NN_BCN_angles"]) for i in low_energy_states],
            c="tab:blue",
            s=20,
            ec="none",
            label="low energy",
        )

    for i in nnmaps[ligand_name]:
        ax.axvline(x=i, c="k", alpha=0.4, ls="--")
    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_ylabel("sum binder angles [deg]", fontsize=16)
    ax.set_ylim(0, 360)
    ax.set_xlabel("N-N distance [AA]", fontsize=16)
    ax.set_xlim(0, 30)
    ax.legend(fontsize=16)

    fig.tight_layout()
    fig.savefig(
        figures_dir / f"cs1_{ligand_name}.png",
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()


def plot_flexes(ligand_db: atomlite.Database, figures_dir: pathlib.Path) -> None:
    fig, axs = plt.subplots(ncols=2, figsize=(16, 5))
    ax, ax1 = axs
    all_xs = []
    all_ys = []
    all_strs = []
    all_nrotatables = []
    all_shuhei_length = []
    for entry in ligand_db.get_entries():
        conf_data = entry.properties["conf_data"]
        min_energy = min([conf_data[i]["UFFEnergy;kj/mol"] for i in conf_data])
        low_energy_states = [
            i
            for i in conf_data
            if (conf_data[i]["UFFEnergy;kj/mol"] - min_energy)
            < EnvVariables.strain_cutoff
        ]
        sigma_distances = np.std(
            [conf_data[i]["NN_distance"] for i in low_energy_states]
        )
        sigma_angles = np.std(
            [sum(conf_data[i]["NN_BCN_angles"]) for i in low_energy_states]
        )
        all_xs.append(sigma_distances)
        all_ys.append(sigma_angles)
        all_strs.append(entry.key)
        rdkit_mol = atomlite.json_to_rdkit(entry.molecule)
        rdmolops.SanitizeMol(rdkit_mol)
        num_rotatable_bonds = rdMolDescriptors.CalcNumRotatableBonds(
            rdkit_mol, rdMolDescriptors.NumRotatableBondsOptions.NonStrict
        ) + get_num_alkynes(rdkit_mol)
        all_nrotatables.append(num_rotatable_bonds)

        new_mol = stk.BuildingBlock.init_from_rdkit_mol(
            rdkit_mol,
            functional_groups=stko.functional_groups.ThreeSiteFactory(
                smarts="[#6]~[#7X2]~[#6]", bonders=(1,), deleters=()
            ),
        )
        target_atom_ids = tuple(
            list(i.get_bonder_ids())[0] for i in new_mol.get_functional_groups()
        )
        dist_matrix = rdmolops.GetDistanceMatrix(rdkit_mol)
        shuhei_length = dist_matrix[target_atom_ids[0]][target_atom_ids[1]]
        all_shuhei_length.append(shuhei_length)

    ax.scatter(
        all_xs,
        all_ys,
        c="tab:blue",
        s=60,
        ec="k",
    )
    for y, s in zip(all_ys, all_strs):
        ax.text(x=1.0, y=y, s=s, fontsize=16)

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_ylabel("$\sigma$ (sum binder angles) [deg]", fontsize=16)
    ax.set_xlabel("$\sigma$ (N-N distance) [AA]", fontsize=16)
    ax.set_ylim(0, None)
    ax.set_xlim(0, None)

    ax1.scatter(
        all_shuhei_length,
        all_nrotatables,
        c="tab:blue",
        s=60,
        ec="k",
    )
    for x, s in zip(all_shuhei_length, all_strs):
        ax1.text(x=x, y=1.0, s=s, fontsize=16)

    ax1.tick_params(axis="both", which="major", labelsize=16)
    ax1.set_ylabel("num. rotatable bonds", fontsize=16)
    ax1.set_xlabel("graph length", fontsize=16)
    ax1.set_ylim(0, None)
    ax1.set_xlim(0, None)

    fig.tight_layout()
    fig.savefig(
        figures_dir / "cs1_flexes.png",
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()


def main():
    args = _parse_args()
    ligand_dir = pathlib.Path("/home/atarzia/workingspace/cpl/case_study_1")
    calculation_dir = pathlib.Path("/home/atarzia/workingspace/cpl/cs1_calculations")
    figures_dir = pathlib.Path("/home/atarzia/workingspace/cpl/figures/cs1/")
    ligand_dir.mkdir(exist_ok=True)
    calculation_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True, parents=True)

    ligand_db = atomlite.Database(ligand_dir / "case_study_1.db")
    pair_db = atomlite.Database(ligand_dir / "cs1_pairs.db")

    # Build all ligands.
    ligand_smiles = {
        # Converging.
        "lab_0": "C1=CC=NC=C1C1=CC(C#CC2=CC(C(NC3=CN=CC=C3)=O)=CC=C2)=CC=C1",
        # Diverging.
        "la_0": ("C1=NC=C2C=CC=C(C#CC3=CC=CN=C3)C2=C1"),
        "lb_0": (
            "C([H])1C([H])=C2C(C([H])=C([H])C([H])=C2C2=C([H])C([H])=C(C3C([H]"
            ")=NC([H])=C([H])C=3[H])C([H])=C2[H])=C([H])N=1"
        ),
        "lc_0": ("C1=CC(=CN=C1)C#CC2=CN=CC=C2"),
        "ld_0": ("C1=CC(=CN=C1)C2=CC=C(C=C2)C3=CN=CC=C3"),
    }
    for lname in ligand_smiles:
        lowe_file = ligand_dir / f"{lname}_lowe.mol"
        if lowe_file.exists():
            molecule = stk.BuildingBlock.init_from_file(lowe_file)
        else:
            # Build polymer.
            molecule = stk.BuildingBlock(smiles=ligand_smiles[lname])
            molecule = stko.ETKDG().optimize(molecule)

        if not ligand_db.has_entry(key=lname):
            explore_ligand(
                molecule=molecule,
                ligand_name=lname,
                ligand_dir=ligand_dir,
                figures_dir=figures_dir,
                calculation_dir=calculation_dir,
                ligand_db=ligand_db,
            )

        if args.plot_ligands:
            plot_ligand(
                ligand_name=lname,
                ligand_db=ligand_db,
                ligand_dir=ligand_dir,
                figures_dir=figures_dir,
            )

    if args.plot_ligands:
        plot_flexes(ligand_db=ligand_db, figures_dir=figures_dir)

    for ligand1, ligand2 in it.combinations(ligand_smiles, 2):
        logging.info(f"analysing {ligand1} and {ligand2}")
        key = f"{ligand1}_{ligand2}"
        if pair_db.has_property_entry(key):
            continue

        analyse_ligand_pair(
            ligand1=ligand1,
            ligand2=ligand2,
            key=key,
            ligand_db=ligand_db,
            pair_db=pair_db,
        )

        raise SystemExit


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    main()
