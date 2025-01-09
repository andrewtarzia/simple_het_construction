"""Script to build the ligands in this case study."""

import itertools as it
import json
import logging
import pathlib
import time

import atomlite
import bbprep
import matplotlib.pyplot as plt
import numpy as np
import stk
import stko
from rdkit.Chem import AllChem as rdkit  # noqa: N813
from rmsd import kabsch_rmsd

from shc.matching_functions import (
    angle_test,
    mismatch_test,
)
from shc.study_2_case_study_1 import experimental_ligand_outcomes, mean_res_str
from shc.utilities import update_from_rdkit_conf


def analyse_ligand_pair(  # noqa: PLR0913
    ligand1: str,
    ligand2: str,
    key: str,
    ligand_db: atomlite.Database,
    pair_db: atomlite.Database,
    dihedral_cutoff: float,
    strain_cutoff: float,
    k_angle: float,
    k_bond: float,
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
    # Iterate over the product of all conformers.
    for cid_1, cid_2 in it.product(ligand1_confs, ligand2_confs):
        cid_name = f"{cid_1}-{cid_2}"

        num_pairs += 1
        # Check strain.
        strain1 = (
            ligand1_confs[cid_1]["UFFEnergy;kj/mol"]
            - ligand1_entry.properties["min_energy;kj/mol"]
        )
        strain2 = (
            ligand2_confs[cid_2]["UFFEnergy;kj/mol"]
            - ligand2_entry.properties["min_energy;kj/mol"]
        )
        if strain1 > strain_cutoff or strain2 > strain_cutoff:
            continue

        # Check torsion.
        torsion1 = abs(ligand1_confs[cid_1]["NCCN_dihedral"])
        torsion2 = abs(ligand2_confs[cid_2]["NCCN_dihedral"])
        if torsion1 > dihedral_cutoff or torsion2 > dihedral_cutoff:
            continue

        # Calculate geom score for both sides together.
        c_dict1 = ligand1_confs[cid_1]
        c_dict2 = ligand2_confs[cid_2]

        # Calculate final geometrical properties.
        angle_dev = angle_test(c_dict1=c_dict1, c_dict2=c_dict2)

        pair_results = mismatch_test(
            c_dict1=c_dict1,
            c_dict2=c_dict2,
            k_angle=k_angle,
            k_bond=k_bond,
        )

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

    return num_pairs_passed


def k_angle_plot(  # noqa: PLR0913, C901
    plot_targets: tuple[tuple[str, str], ...],
    rmsds: list[float],
    strains: list[float],
    dihedrals: list[float],
    k_bonds: list[float],
    k_angles: list[float],
    figures_dir: pathlib.Path,
    ligand_dir: pathlib.Path,
) -> None:
    """Make plot."""
    fig, ax1 = plt.subplots(ncols=1, nrows=1, sharey=True, figsize=(8, 5))

    t_rmsd_threshold = 0.2
    t_dihedral_cutoff = 10
    t_strain_cutoff = 5
    t_k_bond = 1

    skips = ("lab_0_la_0", "e11_0_e14_0", "e12_0_e14_0", "e11_0_e13_0")

    data = {}
    for i, (
        rmsd_threshold,
        dihedral_cutoff,
        strain_cutoff,
        k_angle,
        k_bond,
    ) in enumerate(it.product(rmsds, dihedrals, strains, k_angles, k_bonds)):
        if rmsd_threshold != t_rmsd_threshold:
            continue
        if dihedral_cutoff != t_dihedral_cutoff:
            continue
        if strain_cutoff != t_strain_cutoff:
            continue
        if k_bond != t_k_bond:
            continue

        prefix = (
            f"{i}_{rmsd_threshold}_{dihedral_cutoff}_{strain_cutoff}_"
            f"{k_angle}_{k_bond}"
        )
        pair_db = atomlite.Database(ligand_dir / f"cs1hp_p{prefix}.db")

        for ligand1, ligand2 in plot_targets:
            key = f"{ligand1}_{ligand2}"
            if key in skips:
                continue

            entry = pair_db.get_property_entry(key)

            if entry is None:
                continue

            xdata = [
                entry.properties["pair_data"][i]["state_1_residual"]
                for i in entry.properties["pair_data"]
            ]

            if experimental_ligand_outcomes[(ligand1, ligand2)]["success"]:
                col = "tab:blue"

            else:
                col = "tab:orange"

            if key not in data:
                data[key] = []
            data[key].append((k_angle, np.mean(xdata), col))

    for lkey, ldata in data.items():
        ax1.plot(
            [i[0] for i in ldata],
            [i[1] for i in ldata],
            marker="o" if ldata[0][2] == "tab:blue" else "X",
            markersize=8,
            mec="k",
            label=lkey,
        )

    ax1.tick_params(axis="both", which="major", labelsize=16)
    ax1.set_ylim(0, None)
    ax1.set_ylabel(rf"{mean_res_str}", fontsize=16)
    ax1.set_xlabel(r"$k_{\mathrm{angle}}$", fontsize=16)

    ax1.legend(fontsize=16)

    fig.tight_layout()
    fig.savefig(
        figures_dir / "hp_kangle.png",
        dpi=360,
        bbox_inches="tight",
    )
    plt.close()


def strain_plot(  # noqa: PLR0913, C901
    plot_targets: tuple[tuple[str, str], ...],
    rmsds: list[float],
    strains: list[float],
    dihedrals: list[float],
    k_bonds: list[float],
    k_angles: list[float],
    figures_dir: pathlib.Path,
    ligand_dir: pathlib.Path,
) -> None:
    """Make plot."""
    fig, ax1 = plt.subplots(ncols=1, nrows=1, sharey=True, figsize=(8, 5))

    t_rmsd_threshold = 0.2
    t_dihedral_cutoff = 10
    t_k_bond = 1
    t_k_angle = 5

    skips = ("lab_0_la_0", "e11_0_e14_0", "e12_0_e14_0", "e11_0_e13_0")

    data = {}
    for i, (
        rmsd_threshold,
        dihedral_cutoff,
        strain_cutoff,
        k_angle,
        k_bond,
    ) in enumerate(it.product(rmsds, dihedrals, strains, k_angles, k_bonds)):
        if rmsd_threshold != t_rmsd_threshold:
            continue
        if dihedral_cutoff != t_dihedral_cutoff:
            continue
        if k_angle != t_k_angle:
            continue
        if k_bond != t_k_bond:
            continue

        prefix = (
            f"{i}_{rmsd_threshold}_{dihedral_cutoff}_{strain_cutoff}_"
            f"{k_angle}_{k_bond}"
        )
        pair_db = atomlite.Database(ligand_dir / f"cs1hp_p{prefix}.db")

        for ligand1, ligand2 in plot_targets:
            key = f"{ligand1}_{ligand2}"
            if key in skips:
                continue

            entry = pair_db.get_property_entry(key)

            if entry is None:
                continue

            xdata = [
                entry.properties["pair_data"][i]["state_1_residual"]
                for i in entry.properties["pair_data"]
            ]

            if experimental_ligand_outcomes[(ligand1, ligand2)]["success"]:
                col = "tab:blue"

            else:
                col = "tab:orange"

            if key not in data:
                data[key] = []
            data[key].append((strain_cutoff, np.mean(xdata), col))

    for lkey, ldata in data.items():
        ax1.plot(
            [i[0] for i in ldata],
            [i[1] for i in ldata],
            marker="o" if ldata[0][2] == "tab:blue" else "X",
            markersize=8,
            mec="k",
            label=lkey,
        )

    ax1.tick_params(axis="both", which="major", labelsize=16)
    ax1.set_ylim(0, None)
    ax1.set_ylabel(f"{mean_res_str}", fontsize=16)
    ax1.set_xlabel("strain cutoff", fontsize=16)

    ax1.legend(fontsize=16)

    fig.tight_layout()
    fig.savefig(
        figures_dir / "hp_strain.png",
        dpi=360,
        bbox_inches="tight",
    )
    plt.close()


def rmsd_plot(  # noqa: PLR0913, C901
    plot_targets: tuple[tuple[str, str], ...],
    rmsds: list[float],
    strains: list[float],
    dihedrals: list[float],
    k_bonds: list[float],
    k_angles: list[float],
    figures_dir: pathlib.Path,
    ligand_dir: pathlib.Path,
) -> None:
    """Make plot."""
    fig, ax1 = plt.subplots(ncols=1, nrows=1, sharey=True, figsize=(8, 5))

    t_strain = 5
    t_dihedral_cutoff = 10
    t_k_bond = 1
    t_k_angle = 5

    skips = ("lab_0_la_0", "e11_0_e14_0", "e12_0_e14_0", "e11_0_e13_0")

    data = {}
    for i, (
        rmsd_threshold,
        dihedral_cutoff,
        strain_cutoff,
        k_angle,
        k_bond,
    ) in enumerate(it.product(rmsds, dihedrals, strains, k_angles, k_bonds)):
        if strain_cutoff != t_strain:
            continue
        if dihedral_cutoff != t_dihedral_cutoff:
            continue
        if k_angle != t_k_angle:
            continue
        if k_bond != t_k_bond:
            continue

        prefix = (
            f"{i}_{rmsd_threshold}_{dihedral_cutoff}_{strain_cutoff}_"
            f"{k_angle}_{k_bond}"
        )
        pair_db = atomlite.Database(ligand_dir / f"cs1hp_p{prefix}.db")

        for ligand1, ligand2 in plot_targets:
            key = f"{ligand1}_{ligand2}"
            if key in skips:
                continue

            entry = pair_db.get_property_entry(key)

            if entry is None:
                continue

            xdata = [
                entry.properties["pair_data"][i]["state_1_residual"]
                for i in entry.properties["pair_data"]
            ]

            if experimental_ligand_outcomes[(ligand1, ligand2)]["success"]:
                col = "tab:blue"

            else:
                col = "tab:orange"

            if key not in data:
                data[key] = []
            data[key].append((rmsd_threshold, np.mean(xdata), col))

    for lkey, ldata in data.items():
        ax1.plot(
            [i[0] for i in ldata],
            [i[1] for i in ldata],
            marker="o" if ldata[0][2] == "tab:blue" else "X",
            markersize=8,
            mec="k",
            label=lkey,
        )

    ax1.tick_params(axis="both", which="major", labelsize=16)
    ax1.set_ylim(0, None)
    ax1.set_ylabel(f"{mean_res_str}", fontsize=16)
    ax1.set_xlabel("rmsd cutoff", fontsize=16)

    ax1.legend(fontsize=16)

    fig.tight_layout()
    fig.savefig(
        figures_dir / "hp_rmsd.png",
        dpi=360,
        bbox_inches="tight",
    )
    plt.close()


def dihedral_plot(  # noqa: PLR0913, C901
    plot_targets: tuple[tuple[str, str], ...],
    rmsds: list[float],
    strains: list[float],
    dihedrals: list[float],
    k_bonds: list[float],
    k_angles: list[float],
    figures_dir: pathlib.Path,
    ligand_dir: pathlib.Path,
) -> None:
    """Make plot."""
    fig, ax1 = plt.subplots(ncols=1, nrows=1, sharey=True, figsize=(8, 5))

    t_rmsd_threshold = 0.2
    t_strain = 5
    t_k_bond = 1
    t_k_angle = 5

    skips = ()

    data = {}
    for i, (
        rmsd_threshold,
        dihedral_cutoff,
        strain_cutoff,
        k_angle,
        k_bond,
    ) in enumerate(it.product(rmsds, dihedrals, strains, k_angles, k_bonds)):
        if rmsd_threshold != t_rmsd_threshold:
            continue
        if strain_cutoff != t_strain:
            continue
        if k_angle != t_k_angle:
            continue
        if k_bond != t_k_bond:
            continue

        prefix = (
            f"{i}_{rmsd_threshold}_{dihedral_cutoff}_{strain_cutoff}_"
            f"{k_angle}_{k_bond}"
        )
        pair_db = atomlite.Database(ligand_dir / f"cs1hp_p{prefix}.db")

        for ligand1, ligand2 in plot_targets:
            key = f"{ligand1}_{ligand2}"
            if key in skips:
                continue

            entry = pair_db.get_property_entry(key)

            if entry is None:
                continue

            xdata = [
                entry.properties["pair_data"][i]["state_1_residual"]
                for i in entry.properties["pair_data"]
            ]

            if experimental_ligand_outcomes[(ligand1, ligand2)]["success"]:
                col = "tab:blue"

            else:
                col = "tab:orange"

            if key not in data:
                data[key] = []
            data[key].append((dihedral_cutoff, np.mean(xdata), col))

    for lkey, ldata in data.items():
        ax1.plot(
            [i[0] for i in ldata],
            [i[1] for i in ldata],
            marker="o" if ldata[0][2] == "tab:blue" else "X",
            markersize=8,
            mec="k",
            label=lkey,
        )

    ax1.tick_params(axis="both", which="major", labelsize=16)
    ax1.set_ylim(0, None)
    ax1.set_ylabel(f"{mean_res_str}", fontsize=16)
    ax1.set_xlabel("dihedral cutoff", fontsize=16)

    ax1.legend(fontsize=16)

    fig.tight_layout()
    fig.savefig(
        figures_dir / "hp_dihedral.png",
        dpi=360,
        bbox_inches="tight",
    )
    plt.close()


def explore_ligand(
    molecule: stk.Molecule,
    ligand_name: str,
    ligand_dir: pathlib.Path,
    ligand_db: atomlite.Database,
    rmsd_threshold: float,
) -> None:
    """Do conformer scan."""
    st = time.time()
    conf_dir = ligand_dir / f"confs_{ligand_name}"
    conf_dir.mkdir(exist_ok=True)

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
        conf_opt_file_name = f"{ligand_name}_c{cid}_cuff.mol"

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
                if min_rmsd < rmsd_threshold:
                    break

        # If any RMSD is less than threshold, skip.
        if min_rmsd < rmsd_threshold:
            continue

        new_mol.write(conf_dir / conf_opt_file_name)
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

    entry = atomlite.Entry.from_rdkit(
        key=ligand_name,
        molecule=stk.BuildingBlock.init_from_file(
            ligand_dir / f"{ligand_name}_lowe.mol"
        ).to_rdkit_mol(),
        properties={
            "conf_data": lig_conf_data,
            "min_energy;kj/mol": min_energy * 4.184,
            "ligand_pattern": ligand_name.split("_")[0],
            "composition_pattern": ligand_name.split("_")[1],
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


def plot_conformer_numbers(  # noqa: PLR0915
    ligand_dir: pathlib.Path,
    figures_dir: pathlib.Path,
    rmsds: list[float],
    strains: list[float],
    dihedrals: list[float],
) -> None:
    """Plot ligand properties."""
    ligands = (
        "lab_0",
        "sl1_0",
        "sl2_0",
        "la_0",
        "sla_0",
        "slc_0",
        "e11_0",
        "e12_0",
        "e13_0",
        "e14_0",
    )

    fig, axs = plt.subplots(
        ncols=5,
        nrows=2,
        sharex=True,
        sharey=True,
        figsize=(16, 10),
    )
    flat_axs = axs.flatten()
    for lig, ax in zip(ligands, flat_axs, strict=False):
        for i, (rmsd, strain, dihedral) in enumerate(
            it.product(rmsds, strains, dihedrals)
        ):
            # RMSD only impacts here, so do not rerun conformer db for all
            # cases.
            rmsd_prefix = f"r{rmsd}"
            ligand_db = atomlite.Database(
                ligand_dir / f"cs1hp_{rmsd_prefix}.db"
            )
            for entry in ligand_db.get_entries():
                if entry.key != lig:
                    continue

                original_number = 500
                conf_data = entry.properties["conf_data"]
                after_rmsd = len(conf_data)
                min_energy = min(
                    [conf_data[i]["UFFEnergy;kj/mol"] for i in conf_data]
                )
                low_energy_states = [
                    i
                    for i in conf_data
                    if (conf_data[i]["UFFEnergy;kj/mol"] - min_energy) < strain
                ]

                after_strain = len(low_energy_states)

                untwisted_states = [
                    i
                    for i in low_energy_states
                    if abs(conf_data[i]["NCCN_dihedral"]) <= dihedral
                ]
                after_torsion = len(untwisted_states)

                if rmsd == 0.2:  # noqa: PLR2004
                    c = "tab:blue"
                elif rmsd == 0.5:  # noqa: PLR2004
                    c = "tab:orange"
                elif rmsd == 0.05:  # noqa: PLR2004
                    c = "tab:green"

                ax.plot(
                    [0, 1, 2, 3],
                    [original_number, after_rmsd, after_strain, after_torsion],
                    lw=2,
                    c=c,
                    marker="o",
                    markerfacecolor=c,
                    markersize=6,
                    mec="none",
                    label=f"{dihedral}deg|{strain}kJmol-1|{rmsd}A",
                )

        ax.tick_params(axis="both", which="major", labelsize=16)
        ax.set_xlabel("screening stage", fontsize=16)
        ax.set_ylabel("number conformers", fontsize=16)
        ax.set_xticks([0, 1, 2, 3])
        ax.set_title(lig, fontsize=16)
        ax.set_xticklabels(["initial", "RMSD", "strain", "torsion"])
        ax.set_yscale("log")

    fig.tight_layout()
    fig.savefig(
        figures_dir / "hp_conformer_numbers.png",
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()

    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, sharey=True, figsize=(16, 5))

    for i, (rmsd, strain, dihedral) in enumerate(
        it.product(rmsds, strains, dihedrals)
    ):
        # RMSD only impacts here, so do not rerun conformer db for all cases.
        rmsd_prefix = f"r{rmsd}"
        ligand_db = atomlite.Database(ligand_dir / f"cs1hp_{rmsd_prefix}.db")
        for entry in ligand_db.get_entries():
            original_number = 500
            conf_data = entry.properties["conf_data"]
            after_rmsd = len(conf_data)
            min_energy = min(
                [conf_data[i]["UFFEnergy;kj/mol"] for i in conf_data]
            )
            low_energy_states = [
                i
                for i in conf_data
                if (conf_data[i]["UFFEnergy;kj/mol"] - min_energy) < strain
            ]

            untwisted_states = [
                i
                for i in low_energy_states
                if abs(conf_data[i]["NCCN_dihedral"]) <= dihedral
            ]
            final = len(untwisted_states)

            ax1.scatter(rmsd, final, ec="k", c="tab:blue", s=80)
            ax2.scatter(strain, final, ec="k", c="tab:blue", s=80)
            ax3.scatter(dihedral, final, ec="k", c="tab:blue", s=80)

    ax1.tick_params(axis="both", which="major", labelsize=16)
    ax1.set_xlabel("RMSD threshold", fontsize=16)
    ax1.set_ylabel("final number conformers", fontsize=16)

    ax2.tick_params(axis="both", which="major", labelsize=16)
    ax2.set_xlabel("strain threshold", fontsize=16)

    ax3.tick_params(axis="both", which="major", labelsize=16)
    ax3.set_xlabel("dihedral threshold", fontsize=16)

    fig.tight_layout()
    fig.savefig(
        figures_dir / "hp_conformer_finals.png",
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()


def main() -> None:  # noqa: PLR0915
    """Run script."""
    ligand_dir = pathlib.Path("/home/atarzia/workingspace/cpl/cs1hp")
    calculation_dir = pathlib.Path(
        "/home/atarzia/workingspace/cpl/cs1_calculations"
    )
    figures_dir = pathlib.Path(
        "/home/atarzia/workingspace/cpl/figures/cs1_hp/"
    )
    ligand_dir.mkdir(exist_ok=True)
    calculation_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True, parents=True)

    # Build all ligands.
    ligand_smiles = {
        ##### Converging. #####
        # From Chand.
        "lab_0": "C1=CC=NC=C1C1=CC(C#CC2=CC(C(NC3=CN=CC=C3)=O)=CC=C2)=CC=C1",
        # From study 1.
        "sl1_0": "C1=NC=CC(C2=CC=C3OC4C=CC(C5C=CN=CC=5)=CC=4C3=C2)=C1",
        "sl2_0": "C1=CC(=CC(=C1)C2=CC=NC=C2)C3=CC=NC=C3",
        # From Chand.
        "la_0": ("C1=NC=C2C=CC=C(C#CC3=CC=CN=C3)C2=C1"),
        # From study 1.
        "sla_0": (
            "C1=CN=CC2C(C3=CC=C(C#CC4=CC5C6C=C(C#CC7=CC=C(C8=CC=CC9C=C"
            "N=CC8=9)C=C7)C=CC=6OC=5C=C4)C=C3)=CC=CC1=2"
        ),
        "slc_0": (
            "C1C2=C(C(=CC=C2)C2C=CC(C#CC3=CC=CC(C#CC4C=CC(C5C6=C(C=CN="
            "C6)C=CC=5)=CC=4)=C3)=CC=2)C=NC=1"
        ),
        # Experimental.
        "e11_0": "C1N=CC=CC=1C1=CC2=C(C3=C(C2(C)C)C=C(C2=CN=CC=C2)C=C3)C=C1",
        "e12_0": "C1=CC=C(C2=CC3C(=O)C4C=C(C5=CN=CC=C5)C=CC=4C=3C=C2)C=N1",
        "e13_0": (
            "C1C=C(N2C(=O)C3=C(C=C4C(=C3)C3(C5=C(C4(C)CC3)C=C3C(C(N(C3="
            "O)C3C=CC=NC=3)=O)=C5)C)C2=O)C=NC=1"
        ),
        "e14_0": (
            "C1=CN=CC(C#CC2C=CC3C(=O)C4C=CC(C#CC5=CC=CN=C5)=CC=4C=3C=2)=C1"
        ),
    }

    timings = figures_dir / "timings.json"
    if timings.exists():
        with timings.open("r") as f:
            times = json.load(f)
    else:
        times = {}

    rmsds = [0.2, 0.5, 0.05]
    dihedrals = [10, 20]
    strains = [5, 1, 3]
    k_angles = [1, 2, 5]
    k_bonds = [1, 2, 5]

    times = {}
    for i, (
        rmsd_threshold,
        dihedral_cutoff,
        strain_cutoff,
        k_angle,
        k_bond,
    ) in enumerate(it.product(rmsds, dihedrals, strains, k_angles, k_bonds)):
        rmsd_prefix = f"r{rmsd_threshold}"
        prefix = (
            f"{i}_{rmsd_threshold}_{dihedral_cutoff}_{strain_cutoff}_"
            f"{k_angle}_{k_bond}"
        )
        logging.info("doing %s", prefix)

        # RMSD only impacts here, so do not rerun conformer db for all cases.
        ligand_db = atomlite.Database(ligand_dir / f"cs1hp_{rmsd_prefix}.db")

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
                    ligand_db=ligand_db,
                    rmsd_threshold=rmsd_threshold,
                )

        pair_db = atomlite.Database(ligand_dir / f"cs1hp_p{prefix}.db")

        targets = (
            ("sla_0", "sl1_0"),
            ("slc_0", "sl1_0"),
            ("sla_0", "sl2_0"),
            ("slc_0", "sl2_0"),
            ("lab_0", "la_0"),
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
            st = time.time()
            num_passed = analyse_ligand_pair(
                ligand1=ligand1,
                ligand2=ligand2,
                key=key,
                ligand_db=ligand_db,
                pair_db=pair_db,
                dihedral_cutoff=dihedral_cutoff,
                strain_cutoff=strain_cutoff,
                k_angle=k_angle,
                k_bond=k_bond,
            )
            if prefix not in times:
                times[prefix] = {}

            if key not in times[prefix]:
                times[prefix][key] = (time.time() - st, num_passed)
            with timings.open("w") as f:
                json.dump(times, f)

    with timings.open("w") as f:
        json.dump(times, f)

    plot_targets_sets = {
        "hp": (
            ("sla_0", "sl1_0"),
            ("slc_0", "sl1_0"),
            ("sla_0", "sl2_0"),
            ("slc_0", "sl2_0"),
            ("lab_0", "la_0"),
            ("e11_0", "e14_0"),
            ("e12_0", "e14_0"),
            ("e11_0", "e13_0"),
            ("e12_0", "e13_0"),
            ("e13_0", "e14_0"),
            ("e11_0", "e12_0"),
        )
    }
    plot_targets = plot_targets_sets["hp"]

    k_angle_plot(
        plot_targets=plot_targets,
        figures_dir=figures_dir,
        ligand_dir=ligand_dir,
        rmsds=rmsds,
        dihedrals=dihedrals,
        strains=strains,
        k_angles=k_angles,
        k_bonds=k_bonds,
    )
    strain_plot(
        plot_targets=plot_targets,
        figures_dir=figures_dir,
        ligand_dir=ligand_dir,
        rmsds=rmsds,
        dihedrals=dihedrals,
        strains=strains,
        k_angles=k_angles,
        k_bonds=k_bonds,
    )
    rmsd_plot(
        plot_targets=plot_targets,
        figures_dir=figures_dir,
        ligand_dir=ligand_dir,
        rmsds=rmsds,
        dihedrals=dihedrals,
        strains=strains,
        k_angles=k_angles,
        k_bonds=k_bonds,
    )
    dihedral_plot(
        plot_targets=plot_targets,
        figures_dir=figures_dir,
        ligand_dir=ligand_dir,
        rmsds=rmsds,
        dihedrals=dihedrals,
        strains=strains,
        k_angles=k_angles,
        k_bonds=k_bonds,
    )

    plot_conformer_numbers(
        figures_dir=figures_dir,
        ligand_dir=ligand_dir,
        rmsds=rmsds,
        dihedrals=dihedrals,
        strains=strains,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    main()
