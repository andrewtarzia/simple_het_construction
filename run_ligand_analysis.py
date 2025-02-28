#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Script to build the ligand in this project.

Author: Andrew Tarzia

"""

import logging
import sys
from rdkit.Chem import Draw

import os
import json
import stk
import stko
from rmsd import kabsch_rmsd
import numpy as np
from rdkit.Chem import AllChem as rdkit
import itertools
import time

from env_set import liga_path
import plotting
from utilities import (
    AromaticCNCFactory,
    update_from_rdkit_conf,
    calculate_N_centroid_N_angle,
    calculate_NN_distance,
    calculate_NN_BCN_angles,
    calculate_NCCN_dihedral,
    get_furthest_pair_FGs,
)


def vector_length():
    """
    Mean value of bond distance to use in candidate selection.

    """
    return 2.02


def get_test_1(large_c_dict, small_c_dict):
    # l_angle = (large_c_dict["NN_BCN_angles"]["NN_BCN1"] - 90) + (
    #     large_c_dict["NN_BCN_angles"]["NN_BCN2"] - 90
    # )

    # s_angle = small_c_dict["bite_angle"]
    # # Version 1
    # return s_angle / l_angle
    # # Version 2
    # return (s_angle + l_angle) / 180

    # Version 3.
    # 180 - angle, to make it the angle toward the binding interaction.
    # E.g. To become internal angle of trapezoid.
    l_angle_1 = 180 - large_c_dict["NN_BCN_angles"]["NN_BCN1"]
    l_angle_2 = 180 - large_c_dict["NN_BCN_angles"]["NN_BCN2"]

    s_angle_1 = 180 - small_c_dict["NN_BCN_angles"]["NN_BCN1"]
    s_angle_2 = 180 - small_c_dict["NN_BCN_angles"]["NN_BCN2"]

    interior_angles = l_angle_1 + l_angle_2 + s_angle_1 + s_angle_2
    return interior_angles / 360


def get_test_2(large_c_dict, small_c_dict, pdn_distance):
    sNN_dist = small_c_dict["NN_distance"]
    lNN_dist = large_c_dict["NN_distance"]
    # 180 - angle, to make it the angle toward the binding interaction.
    # E.g. To become internal angle of trapezoid.
    s_angle1 = 180 - small_c_dict["NN_BCN_angles"]["NN_BCN1"]
    s_angle2 = 180 - small_c_dict["NN_BCN_angles"]["NN_BCN2"]

    bonding_vector_length = 2 * pdn_distance
    se1 = bonding_vector_length * np.sin(np.radians(s_angle1))
    se2 = bonding_vector_length * np.sin(np.radians(s_angle2))

    ideal_dist = sNN_dist + se1 + se2
    return lNN_dist / ideal_dist


def test_N_N_lengths(large_c_dict, small_c_dict):
    large_NN_distance = large_c_dict["NN_distance"]
    small_NN_distance = small_c_dict["NN_distance"]
    if large_NN_distance < small_NN_distance:
        raise ValueError(
            f"large NN ({large_NN_distance}) < small NN "
            f"({small_NN_distance}) distance"
        )


def conformer_generation_uff(
    molecule,
    name,
    lowe_output,
    conf_data_file,
):
    """
    Build a large conformer ensemble with UFF optimisation.

    """
    rmsd_threshold = 0.2
    logging.info(f"building conformer ensemble of {name}")

    confs = molecule.to_rdkit_mol()
    etkdg = rdkit.srETKDGv3()
    etkdg.randomSeed = 1000
    cids = rdkit.EmbedMultipleConfs(
        mol=confs,
        numConfs=500,
        params=etkdg,
    )

    lig_conf_data = {}
    num_confs_generated = 0
    num_confs_kept = 0
    conformers_kept = []
    for cid in cids:
        conf_opt_file_name = str(lowe_output).replace(
            "_lowe.mol", f"_c{cid}_cuff.mol"
        )
        # Update stk_mol to conformer geometry.
        new_mol = update_from_rdkit_conf(
            stk_mol=molecule,
            rdk_mol=confs,
            conf_id=cid,
        )
        # Need to define the functional groups.
        new_mol = stk.BuildingBlock.init_from_molecule(
            molecule=new_mol,
            functional_groups=[AromaticCNCFactory()],
        )
        # Only get two FGs.
        new_mol = new_mol.with_functional_groups(
            functional_groups=get_furthest_pair_FGs(new_mol),
        )

        new_mol = stko.UFF().optimize(mol=new_mol)
        energy = stko.UFFEnergy().get_energy(new_mol)
        num_confs_generated += 1

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

        new_mol.write(conf_opt_file_name)
        conformers_kept.append((cid, new_mol))
        NCCN_dihedral = abs(calculate_NCCN_dihedral(new_mol))
        angle = calculate_N_centroid_N_angle(new_mol)

        lig_conf_data[cid] = {
            "NcentroidN_angle": angle,
            "NCCN_dihedral": NCCN_dihedral,
            "NN_distance": calculate_NN_distance(new_mol),
            "NN_BCN_angles": calculate_NN_BCN_angles(new_mol),
            "UFFEnergy;kj/mol": energy * 4.184,
        }
        num_confs_kept += 1

    logging.info(
        "%s conformers generated for %s, kept %s",
        num_confs_generated,
        name,
        num_confs_kept,
    )

    with open(conf_data_file, "w") as f:
        json.dump(lig_conf_data, f)


def ligand_smiles():
    return {
        # Diverging.
        "l1": "C1=NC=CC(C2=CC=C3OC4C=CC(C5C=CN=CC=5)=CC=4C3=C2)=C1",
        "l2": "C1=CC(=CC(=C1)C2=CC=NC=C2)C3=CC=NC=C3",
        "l3": "C1=CN=CC=C1C2=CC=C(S2)C3=CC=NC=C3",
        # Converging.
        "la": (
            "C1=CN=CC2C(C3=CC=C(C#CC4=CC5C6C=C(C#CC7=CC=C(C8=CC=CC9C=C"
            "N=CC8=9)C=C7)C=CC=6OC=5C=C4)C=C3)=CC=CC1=2"
        ),
        "lb": (
            "C1=CN=CC2C(C3=CC=C(C#CC4N=C(C#CC5=CC=C(C6=CC=CC7C=CN=CC6="
            "7)C=C5)C=CC=4)C=C3)=CC=CC1=2"
        ),
        "lc": (
            "C1C2=C(C(=CC=C2)C2C=CC(C#CC3=CC=CC(C#CC4C=CC(C5C6=C(C=CN="
            "C6)C=CC=5)=CC=4)=C3)=CC=2)C=NC=1"
        ),
        "ld": (
            "C1C2=C(C(=CC=C2)C2C=CC(C#CC3=CC=C(C#CC4C=CC(C5C6=C(C=CN=C"
            "6)C=CC=5)=CC=4)S3)=CC=2)C=NC=1"
        ),
        # Experimental.
        "e10": (
            "C1=CC(C#CC2=CC3C4C=C(C#CC5=CC=CN=C5)C=CC=4N(C)C=3C=C2)=CN=C1"
        ),
        "e11": "C1N=CC=CC=1C1=CC2=C(C3=C(C2(C)C)C=C(C2=CN=CC=C2)C=C3)C=C1",
        "e12": "C1=CC=C(C2=CC3C(=O)C4C=C(C5=CN=CC=C5)C=CC=4C=3C=C2)C=N1",
        "e13": (
            "C1C=C(N2C(=O)C3=C(C=C4C(=C3)C3(C5=C(C4(C)CC3)C=C3C(C(N(C3="
            "O)C3C=CC=NC=3)=O)=C5)C)C2=O)C=NC=1"
        ),
        "e14": (
            "C1=CN=CC(C#CC2C=CC3C(=O)C4C=CC(C#CC5=CC=CN=C5)=CC=4C=3C=2)=C1"
        ),
        "e16": (
            "C(C1=CC2C3C=C(C4=CC=NC=C4)C=CC=3C(OC)=C(OC)C=2C=C1)1=CC=NC=C1"
        ),
        "e17": (
            "C12C=CN=CC=1C(C#CC1=CC=C3C(C(C4=C(N3C)C=CC(C#CC3=CC=CC5C3="
            "CN=CC=5)=C4)=O)=C1)=CC=C2"
        ),
        "e18": (
            "C1(=CC=NC=C1)C#CC1=CC2C3C=C(C#CC4=CC=NC=C4)C=CC=3C(OC)=C(O"
            "C)C=2C=C1"
        ),
    }


def main():
    if not len(sys.argv) == 1:
        logging.info(f"Usage: {__file__}\n" "   Expected 0 arguments:")
        sys.exit()
    else:
        pass

    _wd = liga_path()

    dihedral_cutoff = 10
    strain_cutoff = 5

    yproperties = (
        # "xtb_dmsoenergy",
        "NcentroidN_angle",
        "NN_distance",
        "NCCN_dihedral",
        "NN_BCN_angles",
        "bite_angle",
    )

    lsmiles = ligand_smiles()
    for lig in lsmiles:
        lowe_file = _wd / f"{lig}_lowe.mol"
        confuff_data_file = _wd / f"{lig}_conf_uff_data.json"
        unopt_mol = stk.BuildingBlock(
            smiles=lsmiles[lig],
            functional_groups=(AromaticCNCFactory(),),
        )
        rdkit_mol = rdkit.MolFromSmiles(stk.Smiles().get_key(unopt_mol))
        Draw.MolToFile(rdkit_mol, _wd / f"{lig}_2d.png", size=(300, 300))

        if not os.path.exists(confuff_data_file):
            st = time.time()
            conformer_generation_uff(
                molecule=unopt_mol,
                name=lig,
                lowe_output=lowe_file,
                conf_data_file=confuff_data_file,
            )
            logging.info(
                f"time taken for conf gen of {lig}: "
                f"{round(time.time()-st, 2)}s"
            )

    experimental_ligand_outcomes = {
        # Small, large.
        ("e16", "e10"): "yes",
        ("e16", "e17"): "yes",
        ("e10", "e17"): "no",
        ("e11", "e10"): "yes",
        ("e16", "e14"): "yes",
        ("e18", "e14"): "yes",
        ("e18", "e10"): "yes",
        ("e12", "e10"): "yes",
        ("e11", "e14"): "yes",
        ("e12", "e14"): "yes",
        ("e11", "e13"): "yes",
        ("e12", "e13"): "yes",
        ("e13", "e14"): "no",
        ("e11", "e12"): "no",
        # ("e15", "e14"): "yes",
    }
    ligand_pairings = [
        # Small, large.
        ("l1", "la"),
        ("l1", "lb"),
        ("l1", "lc"),
        ("l1", "ld"),
        ("l2", "la"),
        ("l2", "lb"),
        ("l2", "lc"),
        ("l2", "ld"),
        ("l3", "la"),
        ("l3", "lb"),
        ("l3", "lc"),
        ("l3", "ld"),
    ] + list(experimental_ligand_outcomes.keys())

    res_file = os.path.join(_wd, "all_ligand_res.json")
    pair_file = os.path.join(_wd, "all_pair_res.json")
    conf_data_suffix = "conf_uff_data"
    figure_prefix = "etkdg"

    structure_results = {}
    if os.path.exists(res_file):
        with open(res_file, "r") as f:
            structure_results = json.load(f)
    else:
        for ligand in ligand_smiles():
            st = time.time()
            structure_results[ligand] = {}
            conf_data_file = _wd / f"{ligand}_{conf_data_suffix}.json"
            with open(conf_data_file, "r") as f:
                property_dict = json.load(f)

            for cid in property_dict:
                pdi = property_dict[cid]["NN_BCN_angles"]
                # 180 - angle, to make it the angle toward the binding
                # interaction. Minus 90  to convert to the bite-angle.
                ba = ((180 - pdi["NN_BCN1"]) - 90) + (
                    (180 - pdi["NN_BCN2"]) - 90
                )
                property_dict[cid]["bite_angle"] = ba

            structure_results[ligand] = property_dict

            for yprop in yproperties:
                plotting.plot_single_distribution(
                    results_dict=structure_results[ligand],
                    outname=f"{figure_prefix}_d_{ligand}_{yprop}",
                    yproperty=yprop,
                )

            logging.info(
                f"time taken for getting struct results {ligand}: "
                f"{round(time.time()-st, 2)}s"
            )
        with open(res_file, "w") as f:
            json.dump(structure_results, f, indent=4)

    # Define minimum energies for all ligands.
    low_energy_values = {}
    for ligand in structure_results:
        sres = structure_results[ligand]
        min_energy = 1e24
        min_e_cid = 0
        for cid in sres:
            energy = sres[cid]["UFFEnergy;kj/mol"]
            if energy < min_energy:
                min_energy = energy
                min_e_cid = cid
        low_energy_values[ligand] = (min_e_cid, min_energy)

    if os.path.exists(pair_file):
        logging.info(f"loading {pair_file}")
        with open(pair_file, "r") as f:
            pair_info = json.load(f)
    else:
        pair_info = {}
        min_geom_scores = {}
        for small_l, large_l in ligand_pairings:
            logging.info(f"analysing {small_l} and {large_l}")
            st = time.time()
            min_geom_score = 1e24
            pair_name = ",".join((small_l, large_l))
            pair_info[pair_name] = {}
            small_l_dict = structure_results[small_l]
            large_l_dict = structure_results[large_l]

            # Iterate over the product of all conformers.
            for small_cid, large_cid in itertools.product(
                small_l_dict, large_l_dict
            ):
                cid_name = ",".join((small_cid, large_cid))
                # Calculate geom score for both sides together.
                large_c_dict = large_l_dict[large_cid]
                small_c_dict = small_l_dict[small_cid]

                # Calculate final geometrical properties.
                # T1.
                angle_dev = get_test_1(
                    large_c_dict=large_c_dict,
                    small_c_dict=small_c_dict,
                )
                # T2.
                length_dev = get_test_2(
                    large_c_dict=large_c_dict,
                    small_c_dict=small_c_dict,
                    pdn_distance=vector_length(),
                )
                geom_score = abs(angle_dev - 1) + abs(length_dev - 1)

                small_energy = small_l_dict[small_cid]["UFFEnergy;kj/mol"]
                small_strain = small_energy - low_energy_values[small_l][1]
                large_energy = large_l_dict[large_cid]["UFFEnergy;kj/mol"]
                large_strain = large_energy - low_energy_values[large_l][1]
                if (
                    small_strain > strain_cutoff
                    or large_strain > strain_cutoff
                ):
                    continue
                # total_strain = large_strain + small_strain

                min_geom_score = min((geom_score, min_geom_score))
                pair_info[pair_name][cid_name] = {
                    "geom_score": geom_score,
                    # "swapped_LS": swapped_LS,
                    # "converging": converging,
                    # "diverging": diverging,
                    "large_dihedral": large_c_dict["NCCN_dihedral"],
                    "small_dihedral": small_c_dict["NCCN_dihedral"],
                    "angle_deviation": angle_dev,
                    "length_deviation": length_dev,
                    # "small_NCN_angle": small_c_dict["NcentroidN_angle"],
                    # "large_NCN_angle": large_c_dict["NcentroidN_angle"],
                    # "small_energy": small_energy,
                    # "large_energy": large_energy,
                    # "small_strain": small_strain,
                    # "large_strain": large_strain,
                    # "total_strain": total_strain,
                }
            min_geom_scores[pair_name] = round(min_geom_score, 2)
            ft = time.time()
            logging.info(
                f"time taken for pairing {small_l}, {large_l}: "
                f"{round(1000*(ft-st), 2)}ms "
                f"({round(1000*(ft-st)/len(pair_info[pair_name]), 2)}ms"
                f" per pair) - {len(pair_info[pair_name])} pairs"
            )

        logging.info(f"Min. geom scores for each pair:\n {min_geom_scores}")

        with open(pair_file, "w") as f:
            json.dump(pair_info, f, indent=4)

    # Figure in manuscript.
    plotting.gs_table(results_dict=pair_info, dihedral_cutoff=dihedral_cutoff)

    # Figure in manuscript.
    plotting.plot_all_ligand_pairings_simplified(
        results_dict=pair_info,
        dihedral_cutoff=dihedral_cutoff,
        outname=f"{figure_prefix}_all_lp_simpl.pdf",
    )
    plotting.plot_all_ligand_pairings(
        results_dict=pair_info,
        dihedral_cutoff=dihedral_cutoff,
        outname=f"{figure_prefix}_all_lp.png",
    )
    plotting.plot_all_ligand_pairings_2dhist(
        results_dict=pair_info,
        dihedral_cutoff=dihedral_cutoff,
        outname=f"{figure_prefix}_all_lp_2dhist.png",
    )
    plotting.plot_all_ligand_pairings_2dhist_fig5(
        results_dict=pair_info,
        dihedral_cutoff=dihedral_cutoff,
        outname=f"{figure_prefix}_all_lp_2dhist_fig5.pdf",
    )

    # Figures in SI.
    plotting.plot_all_geom_scores_simplified(
        results_dict=pair_info,
        outname=f"{figure_prefix}_all_pairs_simpl.png",
        dihedral_cutoff=dihedral_cutoff,
        experimental_ligand_outcomes=experimental_ligand_outcomes,
    )
    plotting.plot_all_ligand_pairings_conformers(
        results_dict=pair_info,
        structure_results=structure_results,
        dihedral_cutoff=dihedral_cutoff,
        outname=f"{figure_prefix}_all_lp_conformers.pdf",
    )
    raise SystemExit
    # Figures not in SI.
    for pair_name in pair_info:
        small_l, large_l = pair_name.split(",")
        plotting.plot_ligand_pairing(
            results_dict=pair_info[pair_name],
            dihedral_cutoff=dihedral_cutoff,
            outname=f"{figure_prefix}_lp_{small_l}_{large_l}.png",
        )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    main()
