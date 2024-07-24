"""Script to build the ligand in this project."""

import argparse
import itertools as it
import logging
import pathlib
import time

import atomlite
import matplotlib.pyplot as plt
import numpy as np
import stk
import stko
from definitions import EnvVariables
from matching_functions import angle_test, mismatch_test, plot_pair_position
from rdkit.Chem import AllChem as rdkit  # noqa: N813
from rdkit.Chem import rdMolDescriptors, rdmolops, rdMolTransforms
from study_2_build_ligands import explore_ligand


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--plot_ligands",
        action="store_true",
        help="plot ligand data",
    )

    return parser.parse_args()


def analyse_ligand_pair(  # noqa: PLR0913
    ligand1: str,
    ligand2: str,
    key: str,
    ligand_db: atomlite.Database,
    pair_db: atomlite.Database,
    figures_dir: pathlib.Path,
) -> None:
    """Analyse a pair of ligands."""
    ligand1_entry = ligand_db.get_entry(ligand1)
    ligand1_confs = ligand1_entry.properties["conf_data"]
    ligand2_entry = ligand_db.get_entry(ligand2)
    ligand2_confs = ligand2_entry.properties["conf_data"]
    st = time.time()
    num_pairs = 0
    num_pairs_passed = 0
    pair_data = {}
    best_pair = None
    best_residual = 1e24
    # Iterate over the product of all conformers.
    for cid_1, cid_2 in it.product(ligand1_confs, ligand2_confs):
        cid_name = f"{cid_1}-{cid_2}"

        num_pairs += 1
        # Check strain.
        strain1 = (
            ligand1_confs[cid_1]["UFFEnergy;kj/mol"]
            - ligand1_entry.properties["min_energy;kj/mol"] * 4.184
        )
        strain2 = (
            ligand2_confs[cid_2]["UFFEnergy;kj/mol"]
            - ligand2_entry.properties["min_energy;kj/mol"] * 4.184
        )
        if (
            strain1 > EnvVariables.strain_cutoff
            or strain2 > EnvVariables.strain_cutoff
        ):
            continue

        # Check torsion.
        torsion1 = abs(ligand1_confs[cid_1]["NCCN_dihedral"])
        torsion2 = abs(ligand2_confs[cid_2]["NCCN_dihedral"])
        if (
            torsion1 > EnvVariables.cs1_dihedral_cutoff
            or torsion2 > EnvVariables.cs1_dihedral_cutoff
        ):
            continue

        # Calculate geom score for both sides together.
        c_dict1 = ligand1_confs[cid_1]
        c_dict2 = ligand2_confs[cid_2]

        # Calculate final geometrical properties.
        angle_dev = angle_test(c_dict1=c_dict1, c_dict2=c_dict2)

        pair_results = mismatch_test(c_dict1=c_dict1, c_dict2=c_dict2)

        pair_data[cid_name] = {
            "state_1_residual": float(pair_results.state_1_result),
            "state_2_residual": float(pair_results.state_2_result),
            "state_1_parameters": [
                float(i) for i in pair_results.state_1_parameters
            ],
            "state_2_parameters": [
                float(i) for i in pair_results.state_2_parameters
            ],
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
        num_pairs_passed += 1
        if (
            float(pair_results.state_1_result) < best_residual
            or float(pair_results.state_2_result) < best_residual
        ):
            best_pair = pair_results
            best_residual = min(
                (
                    float(pair_results.state_1_result),
                    float(pair_results.state_2_result),
                )
            )

    logging.info("in future, save this whole dict with comented code.")
    # print("t", pair_db.has_property_entry(key))  # noqa: ERA001
    #   entry = atomlite.PropertyEntry(  key=key,
    #     properties={"pair_data": pair_data}, # noqa: ERA001
    # pair_db.update_properties(entries=entry) # noqa: ERA001
    # print("t2", pair_db.has_property_entry(key)) # noqa: ERA001
    pair_db.connection.commit()
    ft = time.time()
    logging.info(
        "pairing %s, %s: " "%s s " "(%s s" " per pair) - %s pairs passed",
        ligand1,
        ligand2,
        round((ft - st), 2),
        round(1000 * (ft - st) / num_pairs),
        num_pairs_passed,
    )

    plot_pair_position(
        r1=np.array(
            (best_pair.set_parameters[0], best_pair.set_parameters[1])
        ),
        phi1=best_pair.set_parameters[2],
        rigidbody1=best_pair.rigidbody1,
        r2=np.array(
            (best_pair.state_1_parameters[0], best_pair.state_1_parameters[1])
        ),
        phi2=best_pair.state_1_parameters[2],
        rigidbody2=best_pair.rigidbody2,
        r3=np.array(
            (best_pair.state_2_parameters[0], best_pair.state_2_parameters[1])
        ),
        phi3=best_pair.state_2_parameters[2],
        rigidbody3=best_pair.rigidbody3,
        outname=figures_dir / f"best_{key}.png",
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
    """Get the number of alkynes in a molecule."""
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
    """Plot ligand properties."""
    nnmaps = {
        "lab_0": (9.8, 12.8),
        "la_0": (8.7,),
        "lb_0": (11,),
        "lc_0": (8.1,),
        "ld_0": (9.9,),
    }

    fig, (ax, ax1) = plt.subplots(ncols=2, figsize=(16, 5))
    entry = ligand_db.get_entry(ligand_name)
    conf_data = entry.properties["conf_data"]
    logging.info("remove conversion")
    min_energy = entry.properties["min_energy;kj/mol"] * 4.184

    low_energy_states = [
        i
        for i in conf_data
        if (conf_data[i]["UFFEnergy;kj/mol"] - min_energy)
        < EnvVariables.strain_cutoff
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
    ax1.scatter(
        [conf_data[i]["NN_BCN_angles"][0] for i in conf_data],
        [conf_data[i]["NN_BCN_angles"][1] for i in conf_data],
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

            torsion_state = (
                "f" if get_amide_torsions(conf_mol)[0] < 90 else "b"  # noqa: PLR2004
            )

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
        ax1.scatter(
            [conf_data[i]["NN_BCN_angles"][0] for i in states["b"]],
            [conf_data[i]["NN_BCN_angles"][1] for i in states["b"]],
            c="tab:blue",
            s=20,
            ec="none",
            label="low energy, b",
        )
        ax1.scatter(
            [conf_data[i]["NN_BCN_angles"][0] for i in states["f"]],
            [conf_data[i]["NN_BCN_angles"][1] for i in states["f"]],
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
        ax1.scatter(
            [conf_data[i]["NN_BCN_angles"][0] for i in low_energy_states],
            [conf_data[i]["NN_BCN_angles"][1] for i in low_energy_states],
            c="tab:blue",
            s=20,
            ec="none",
            label="low energy",
        )

    if ligand_name in nnmaps:
        for i in nnmaps[ligand_name]:
            ax.axvline(x=i, c="k", alpha=0.4, ls="--")
    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_ylabel("sum binder angles [deg]", fontsize=16)
    ax.set_ylim(0, 360)
    ax.set_xlabel("N-N distance [AA]", fontsize=16)
    ax.set_xlim(0, 30)
    ax.legend(fontsize=16)

    ax1.tick_params(axis="both", which="major", labelsize=16)
    ax1.set_xlabel("binder angle 1 [deg]", fontsize=16)
    ax1.set_xlim(0, 360)
    ax1.set_ylabel("binder angle 2 [deg]", fontsize=16)
    ax1.set_ylim(0, 360)
    ax1.plot((0, 360), (0, 360), c="k", ls="--")
    fig.tight_layout()
    fig.savefig(
        figures_dir / f"cs1_{ligand_name}.png",
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()


def plot_flexes(
    ligand_db: atomlite.Database,
    figures_dir: pathlib.Path,
) -> None:
    """Plot ligand flexibilities."""
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
            next(iter(i.get_bonder_ids()))
            for i in new_mol.get_functional_groups()
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
    for y, s in zip(all_ys, all_strs, strict=False):
        ax.text(x=1.0, y=y, s=s, fontsize=16)

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_ylabel(r"$\sigma$ (sum binder angles) [deg]", fontsize=16)
    ax.set_xlabel(r"$\sigma$ (N-N distance) [AA]", fontsize=16)
    ax.set_ylim(0, None)
    ax.set_xlim(0, None)

    ax1.scatter(
        all_shuhei_length,
        all_nrotatables,
        c="tab:blue",
        s=60,
        ec="k",
    )
    for x, s in zip(all_shuhei_length, all_strs, strict=False):
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


def main() -> None:  # noqa: C901, PLR0912, PLR0915
    """Run script."""
    args = _parse_args()
    ligand_dir = pathlib.Path("/home/atarzia/workingspace/cpl/case_study_1")
    calculation_dir = pathlib.Path(
        "/home/atarzia/workingspace/cpl/cs1_calculations"
    )
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
        # From molinksa.
        "m2h_0": "C1=CC(=CC(=C1)C#CC2=CN=CC=C2)C#CC3=CN=CC=C3",
        # From study 1.
        "sl1_0": "C1=NC=CC(C2=CC=C3OC4C=CC(C5C=CN=CC=5)=CC=4C3=C2)=C1",
        "sl2_0": "C1=CC(=CC(=C1)C2=CC=NC=C2)C3=CC=NC=C3",
        "sl3_0": "C1=CN=CC=C1C2=CC=C(S2)C3=CC=NC=C3",
        # Diverging.
        "la_0": ("C1=NC=C2C=CC=C(C#CC3=CC=CN=C3)C2=C1"),
        "lb_0": (
            "C([H])1C([H])=C2C(C([H])=C([H])C([H])=C2C2=C([H])C([H])=C(C3C([H]"
            ")=NC([H])=C([H])C=3[H])C([H])=C2[H])=C([H])N=1"
        ),
        "lc_0": ("C1=CC(=CN=C1)C#CC2=CN=CC=C2"),
        "ld_0": ("C1=CC(=CN=C1)C2=CC=C(C=C2)C3=CN=CC=C3"),
        # From molinksa.
        "m4q_0": "C1=CC=C2C(=C1)C=C(C=N2)C#CC3=CN=CC=C3",
        "m4p_0": "C1=CN=CC(C#CC2=CN=C(C)C=C2)=C1",
        # From study 1.
        "sla_0": (
            "C1=CN=CC2C(C3=CC=C(C#CC4=CC5C6C=C(C#CC7=CC=C(C8=CC=CC9C=C"
            "N=CC8=9)C=C7)C=CC=6OC=5C=C4)C=C3)=CC=CC1=2"
        ),
        "slb_0": (
            "C1=CN=CC2C(C3=CC=C(C#CC4N=C(C#CC5=CC=C(C6=CC=CC7C=CN=CC6="
            "7)C=C5)C=CC=4)C=C3)=CC=CC1=2"
        ),
        "slc_0": (
            "C1C2=C(C(=CC=C2)C2C=CC(C#CC3=CC=CC(C#CC4C=CC(C5C6=C(C=CN="
            "C6)C=CC=5)=CC=4)=C3)=CC=2)C=NC=1"
        ),
        "sld_0": (
            "C1C2=C(C(=CC=C2)C2C=CC(C#CC3=CC=C(C#CC4C=CC(C5C6=C(C=CN=C"
            "6)C=CC=5)=CC=4)S3)=CC=2)C=NC=1"
        ),
        # Experimental.
        "e10_0": (
            "C1=CC(C#CC2=CC3C4C=C(C#CC5=CC=CN=C5)C=CC=4N(C)C=3C=C2)=CN=C1"
        ),
        "e11_0": "C1N=CC=CC=1C1=CC2=C(C3=C(C2(C)C)C=C(C2=CN=CC=C2)C=C3)C=C1",
        "e12_0": "C1=CC=C(C2=CC3C(=O)C4C=C(C5=CN=CC=C5)C=CC=4C=3C=C2)C=N1",
        "e13_0": (
            "C1C=C(N2C(=O)C3=C(C=C4C(=C3)C3(C5=C(C4(C)CC3)C=C3C(C(N(C3="
            "O)C3C=CC=NC=3)=O)=C5)C)C2=O)C=NC=1"
        ),
        "e14_0": (
            "C1=CN=CC(C#CC2C=CC3C(=O)C4C=CC(C#CC5=CC=CN=C5)=CC=4C=3C=2)=C1"
        ),
        "e16_0": (
            "C(C1=CC2C3C=C(C4=CC=NC=C4)C=CC=3C(OC)=C(OC)C=2C=C1)1=CC=NC=C1"
        ),
        "e17_0": (
            "C12C=CN=CC=1C(C#CC1=CC=C3C(C(C4=C(N3C)C=CC(C#CC3=CC=CC5C3="
            "CN=CC=5)=C4)=O)=C1)=CC=C2"
        ),
        "e18_0": (
            "C1(=CC=NC=C1)C#CC1=CC2C3C=C(C#CC4=CC=NC=C4)C=CC=3C(OC)=C(O"
            "C)C=2C=C1"
        ),
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

    targets = (
        ("sla_0", "sl1_0"),
        ("slb_0", "sl1_0"),
        ("slc_0", "sl1_0"),
        ("sld_0", "sl1_0"),
        ("sla_0", "sl2_0"),
        ("slb_0", "sl2_0"),
        ("slc_0", "sl2_0"),
        ("sld_0", "sl2_0"),
        ("lab_0", "la_0"),
        ("lab_0", "lb_0"),
        ("lab_0", "lc_0"),
        ("lab_0", "ld_0"),
        ("m2h_0", "m4q_0"),
        ("m2h_0", "m4p_0"),
        ("e16_0", "e10_0"),
        ("e16_0", "e17_0"),
        ("e10_0", "e17_0"),
        ("e11_0", "e10_0"),
        ("e16_0", "e14_0"),
        ("e18_0", "e14_0"),
        ("e18_0", "e10_0"),
        ("e12_0", "e10_0"),
        ("e11_0", "e14_0"),
        ("e12_0", "e14_0"),
        ("e11_0", "e13_0"),
        ("e12_0", "e13_0"),
        ("e13_0", "e14_0"),
        ("e11_0", "e12_0"),
    )

    for ligand1, ligand2 in targets:
        key = f"{ligand1}_{ligand2}"

        if pair_db.has_property_entry(key):
            continue

        logging.info("analysing %s and %s", ligand1, ligand2)
        analyse_ligand_pair(
            ligand1=ligand1,
            ligand2=ligand2,
            key=key,
            ligand_db=ligand_db,
            pair_db=pair_db,
            figures_dir=figures_dir,
        )

    plot_targets_sets = {
        "expt": (
            ("e16_0", "e10_0"),
            ("e16_0", "e17_0"),
            ("e10_0", "e17_0"),
            ("e11_0", "e10_0"),
            ("e16_0", "e14_0"),
            ("e18_0", "e14_0"),
            ("e18_0", "e10_0"),
            ("e12_0", "e10_0"),
            ("e11_0", "e14_0"),
            ("e12_0", "e14_0"),
            ("e11_0", "e13_0"),
            ("e12_0", "e13_0"),
            ("e13_0", "e14_0"),
            ("e11_0", "e12_0"),
        ),
        "2024": (
            ("sla_0", "sl1_0"),
            ("slb_0", "sl1_0"),
            ("slc_0", "sl1_0"),
            ("sld_0", "sl1_0"),
            ("sla_0", "sl2_0"),
            ("slb_0", "sl2_0"),
            ("slc_0", "sl2_0"),
            ("sld_0", "sl2_0"),
        ),
        "het": (
            ("lab_0", "la_0"),
            ("lab_0", "lb_0"),
            ("lab_0", "lc_0"),
            ("lab_0", "ld_0"),
            ("m2h_0", "m4q_0"),
            ("m2h_0", "m4p_0"),
        ),
    }
    for pts in plot_targets_sets:
        plot_targets = plot_targets_sets[pts]
        fig, axs = plt.subplots(ncols=3, figsize=(16, 5))
        ax, ax1, ax2 = axs
        steps = range(len(plot_targets) - 1, -1, -1)
        for i, (ligand1, ligand2) in zip(steps, plot_targets, strict=False):
            key = f"{ligand1}_{ligand2}"
            entry = pair_db.get_property_entry(key)

            xdata = [
                entry.properties["pair_data"][i]["state_1_residual"]
                for i in entry.properties["pair_data"]
            ]
            xmin = 0
            xmax = 15
            xwidth = 0.5
            xbins = np.arange(xmin - xwidth, xmax + xwidth, xwidth)
            ystep = 1
            ax.hist(
                x=xdata,
                bins=xbins,
                density=True,
                bottom=i * ystep,
                histtype="stepfilled",
                stacked=True,
                linewidth=1.0,
                alpha=1.0,
                edgecolor="k",
                label=f"{entry.key}",
            )
            ax.plot(
                (np.mean(xdata), np.mean(xdata)),
                ((i + 1) * ystep, i * ystep),
                alpha=1.0,
                c="k",
            )

            xdata = [
                entry.properties["pair_data"][i]["state_2_residual"]
                for i in entry.properties["pair_data"]
            ]
            xmin = 0
            xmax = 15
            xwidth = 0.5
            xbins = np.arange(xmin - xwidth, xmax + xwidth, xwidth)
            ystep = 1
            ax1.hist(
                x=xdata,
                bins=xbins,
                density=True,
                bottom=i * ystep,
                histtype="stepfilled",
                stacked=True,
                linewidth=1.0,
                alpha=1.0,
                edgecolor="k",
                label=f"{entry.key}",
            )
            ax1.plot(
                (np.mean(xdata), np.mean(xdata)),
                ((i + 1) * ystep, i * ystep),
                alpha=1.0,
                c="k",
            )

            xdata = [
                entry.properties["pair_data"][i]["state_1_residual"]
                - entry.properties["pair_data"][i]["state_2_residual"]
                for i in entry.properties["pair_data"]
            ]
            xmin = -1
            xmax = 1
            xwidth = 0.05
            xbins = np.arange(xmin - xwidth, xmax + xwidth, xwidth)
            ystep = 10
            ax2.hist(
                x=xdata,
                bins=xbins,
                density=True,
                bottom=i * ystep,
                histtype="stepfilled",
                stacked=True,
                linewidth=1.0,
                alpha=1.0,
                edgecolor="k",
                label=f"{entry.key}",
            )
            ax2.plot(
                (np.mean(xdata), np.mean(xdata)),
                ((i + 1) * ystep, i * ystep),
                alpha=1.0,
                c="k",
            )

        ax.tick_params(axis="both", which="major", labelsize=16)
        ax.set_xlabel("1-residuals", fontsize=16)
        ax.set_ylabel("frequency", fontsize=16)
        ax.set_yticks([])
        ax.set_ylim(0, (steps[0] + 1.5) * 1)

        ax1.tick_params(axis="both", which="major", labelsize=16)
        ax1.set_xlabel("2-residuals", fontsize=16)
        ax1.set_yticks([])
        ax1.set_ylim(0, (steps[0] + 1.5) * 1)

        ax2.tick_params(axis="both", which="major", labelsize=16)
        ax2.set_xlabel("delta-residuals", fontsize=16)
        ax2.set_yticks([])
        ax2.set_ylim(0, (steps[0] + 1.5) * 10)
        if pts == "expt":
            ax2.legend(ncols=2, fontsize=16)
        else:
            ax2.legend(fontsize=16)

        fig.tight_layout()
        fig.savefig(
            figures_dir / f"cs1_residuals_{pts}.png",
            dpi=360,
            bbox_inches="tight",
        )
        plt.close()

    fig, ax = plt.subplots(figsize=(8, 5))
    plot_targets = (
        ("lab_0", "la_0"),
        ("lab_0", "lb_0"),
        ("lab_0", "lc_0"),
        ("lab_0", "ld_0"),
    )
    steps = range(len(plot_targets) - 1, -1, -1)
    for i, (ligand1, ligand2) in zip(steps, plot_targets, strict=False):
        key = f"{ligand1}_{ligand2}"
        entry = pair_db.get_property_entry(key)
        xmin = 0
        xmax = 15
        xwidth = 0.5
        xbins = np.arange(xmin - xwidth, xmax + xwidth, xwidth)
        ystep = 1

        x1data = []
        x2data = []
        conf_dir = ligand_dir / "confs_lab_0"
        for cid_name in entry.properties["pair_data"]:
            la_conf = cid_name.split("-")[0]
            conf_mol = stk.BuildingBlock.init_from_file(
                conf_dir / f"lab_0_c{la_conf}_cuff.mol"
            )

            torsion_state = (
                "f" if get_amide_torsions(conf_mol)[0] < 90 else "b"  # noqa: PLR2004
            )

            if torsion_state == "f":
                x1data.append(
                    entry.properties["pair_data"][cid_name]["state_1_residual"]
                )
            elif torsion_state == "b":
                x2data.append(
                    entry.properties["pair_data"][cid_name]["state_1_residual"]
                )

        if ligand2 == "la_0":
            lbl1 = "f"
            lbl2 = "b"
        else:
            lbl1 = None
            lbl2 = None

        ax.hist(
            x=x1data,
            bins=xbins,
            density=True,
            bottom=i * ystep,
            histtype="stepfilled",
            stacked=True,
            linewidth=1.0,
            alpha=1.0,
            edgecolor="none",
            label=lbl1,
        )
        ax.plot(
            (np.mean(x1data), np.mean(x1data)),
            ((i + 1) * ystep, i * ystep),
            alpha=1.0,
            c="k",
        )

        ax.hist(
            x=x2data,
            bins=xbins,
            density=True,
            bottom=i * ystep,
            histtype="stepfilled",
            stacked=True,
            linewidth=1.0,
            alpha=1.0,
            edgecolor="k",
            facecolor="none",
            label=lbl2,
        )
        ax.plot(
            (np.mean(x2data), np.mean(x2data)),
            ((i + 1) * ystep, i * ystep),
            alpha=1.0,
            c="r",
        )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel("1-residuals", fontsize=16)
    ax.set_ylabel("frequency", fontsize=16)
    ax.set_yticks([])
    ax.set_ylim(0, (steps[0] + 1.5) * 1)
    ax.legend(fontsize=16)

    fig.tight_layout()
    fig.savefig(
        figures_dir / "lab_state_residuals.png",
        dpi=360,
        bbox_inches="tight",
    )
    plt.close()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    main()
