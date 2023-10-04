#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Script to build the ligand in this project.

Author: Andrew Tarzia

"""

import logging
import sys
import os
import json
import stk
import stko
import bbprep
import numpy as np
from rdkit.Chem import AllChem as rdkit
import itertools
import math
import rmsd

from build_ligands import ligand_smiles
from env_set import liga_path, calc_path
import plotting
from utilities import (
    AromaticCNCFactory,
    update_from_rdkit_conf,
    calculate_N_centroid_N_angle,
    calculate_NN_distance,
    calculate_NN_BCN_angles,
    calculate_NCCN_dihedral,
)


def vector_length():
    """
    Mean value of bond distance to use in candidate selection.

    Ran with 2.05 from CSD survey.

    """
    return 2.05


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


def get_test_2(large_c_dict, small_c_dict):

    sNN_dist = small_c_dict["NN_distance"]
    lNN_dist = large_c_dict["NN_distance"]
    # 180 - angle, to make it the angle toward the binding interaction.
    # E.g. To become internal angle of trapezoid.
    s_angle1 = 180 - small_c_dict["NN_BCN_angles"]["NN_BCN1"]
    s_angle2 = 180 - small_c_dict["NN_BCN_angles"]["NN_BCN2"]

    bonding_vector_length = 2 * vector_length()
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


def get_gs_cutoff(
    results_dict,
    dihedral_cutoff,
    experimental_ligand_outcomes,
):

    max_min_gs = 0
    for pair_name in results_dict:
        rdict = results_dict[pair_name]

        if "e" in pair_name:
            ename = pair_name.split(",")
            if (ename[0], ename[1]) in experimental_ligand_outcomes:
                edata = experimental_ligand_outcomes[(ename[0], ename[1])]

            elif (ename[1], ename[0]) in experimental_ligand_outcomes:
                edata = experimental_ligand_outcomes[(ename[1], ename[0])]
            else:
                continue

            if edata != "yes":
                continue
        else:
            continue

        min_geom_score = 1e24
        for cid_pair in rdict:

            if (
                abs(rdict[cid_pair]["large_dihedral"]) > dihedral_cutoff
                or abs(rdict[cid_pair]["small_dihedral"]) > dihedral_cutoff
            ):
                continue

            geom_score = rdict[cid_pair]["geom_score"]
            if geom_score < min_geom_score:
                min_geom_score = geom_score

        max_min_gs = max([min_geom_score, max_min_gs])

    return max_min_gs


def round_up(n, decimals=0):
    multiplier = 10**decimals
    return math.ceil(n * multiplier) / multiplier


def conformer_generation_uff(
    molecule,
    name,
    lowe_output,
    conf_data_file,
    calc_dir,
):
    """
    Build a large conformer ensemble with UFF optimisation.

    """

    logging.info(f"building conformer ensemble of {name}")

    confs = molecule.to_rdkit_mol()
    etkdg = rdkit.srETKDGv3()
    etkdg.randomSeed = 1000
    etkdg.pruneRmsThresh = 0.2
    cids = rdkit.EmbedMultipleConfs(
        mol=confs,
        numConfs=500,
        params=etkdg,
        # pruneRmsThresh=0.2,
    )

    lig_conf_data = {}
    num_confs = 0
    for cid in cids:
        conf_opt_file_name = str(lowe_output).replace(
            "_lowe.mol", f"_c{cid}_cuff.mol"
        )
        # Update stk_mol to conformer geometry.
        new_mol = update_from_rdkit_conf(
            stk_mol=molecule, rdk_mol=confs, conf_id=cid
        )
        # Need to define the functional groups.
        new_mol = stk.BuildingBlock.init_from_molecule(
            molecule=new_mol,
            functional_groups=[AromaticCNCFactory()],
        )
        # Only get two FGs.
        new_mol = bbprep.FurthestFGs().modify(
            building_block=new_mol,
            desired_functional_groups=2,
        )

        new_mol = stko.UFF().optimize(mol=new_mol)
        energy = stko.UFFEnergy().get_energy(new_mol)
        new_mol.write(conf_opt_file_name)

        NCCN_dihedral = abs(calculate_NCCN_dihedral(new_mol))
        angle = calculate_N_centroid_N_angle(new_mol)

        lig_conf_data[cid] = {
            "NcentroidN_angle": angle,
            "NCCN_dihedral": NCCN_dihedral,
            "NN_distance": calculate_NN_distance(new_mol),
            "NN_BCN_angles": calculate_NN_BCN_angles(new_mol),
            "UFFEnergy;kj/mol": energy * 4.184,
        }
        num_confs += 1
    logging.info(f"{num_confs} conformers generated for {name}")

    with open(conf_data_file, "w") as f:
        json.dump(lig_conf_data, f)


def conformer_generation_scan(
    molecule, name, conf_data_file, dihedral_cutoff, calc_dir
):
    """
    Build a large conformer ensemble with bbprep scan.

    """

    name_to_smarts = {
        "l1": ("[#6X3]@[#6X3H0]-!@[#6X3H0]@[#6X3]", 4, (0, 1, 2, 3)),
        "l2": ("[#6X3]@[#6X3H0]-!@[#6X3H0]@[#6X3]", 4, (0, 1, 2, 3)),
        "l3": ("[#6X3]@[#6X3H0]-!@[#6X3H0]@[#6X3]", 4, (0, 1, 2, 3)),
        "la": ("[#6X3]@[#6X3H0]-!@[#6X3H0]@[#6X3]", 4, (0, 1, 2, 3)),
        "lb": ("[#6X3]@[#6X3H0]-!@[#6X3H0]@[#6X3]", 4, (0, 1, 2, 3)),
        "lc": ("[#6X3]@[#6X3H0]-!@[#6X3H0]@[#6X3]", 4, (0, 1, 2, 3)),
        "ld": ("[#6X3]@[#6X3H0]-!@[#6X3H0]@[#6X3]", 4, (0, 1, 2, 3)),
        "ll1": (
            "[#7X2]@[#6X3]@[#6X3H0]-!@[#6X2H0]#[#6X2H0]-!@[#6X3H0]@[#6X3]",
            7,
            (1, 2, 5, 6),
        ),
        "ls": ("[#6X3]@[#6X3H0]-!@[#6X3H0]@[#6X3]", 4, (0, 1, 2, 3)),
        "ll2": (
            "[#6X3H0]@[#6X3H0]-!@[#6X2H0]#[#6X2H0]-!@[#6X3H0]@[#6X3]",
            6,
            (0, 1, 4, 5),
        ),
        "e1": ("[#6X3]@[#6X3H0]-!@[#6X3H0]@[#6X3]", 4, (0, 1, 2, 3)),
        "e2": (
            "[#7X2]@[#6X3]@[#6X3H0]-!@[#6X2H0]#[#6X2H0]-!@[#6X3H0]@[#6X3]",
            7,
            (1, 2, 5, 6),
        ),
        "e3": ("[#6X3]@[#6X3H0]-!@[#6X3H0]@[#6X3]", 4, (0, 1, 2, 3)),
        "e4": (
            (
                "[#7X2]@[#6X3]@[#6X3]@[#6X3H0]-!@[#6X2H0]#[#6X2H0]-"
                "!@[#6X3H0]@[#6X3]"
            ),
            8,
            (2, 3, 6, 7),
        ),
        "e5": ("[#6X3]@[#6X3H0]-!@[#6X3H0]@[#6X3]", 4, (0, 1, 2, 3)),
        "e6": ("[#6X3]@[#6X3H0]-!@[#6X3H0]@[#6X3]", 4, (0, 1, 2, 3)),
        "e7": (
            (
                "[#7X2]@[#6X3]@[#6X3]@[#6X3H0]-!@[#6X2H0]#[#6X2H0]-"
                "!@[#6X3H0]@[#6X3]"
            ),
            8,
            (2, 3, 6, 7),
        ),
        "e8": ("[#6X3]@[#6X3H0]-!@[#6X3H0]@[#6X3]", 4, (0, 1, 2, 3)),
        "e9": ("[#6X3]@[#6X3H0]-!@[#6X3H0]@[#6X3]", 4, (0, 1, 2, 3)),
        "e10": (
            "[#7X2]@[#6X3]@[#6X3H0]-!@[#6X2H0]#[#6X2H0]-!@[#6X3H0]@[#6X3]",
            7,
            (1, 2, 5, 6),
        ),
        "e11": ("[#6X3]@[#6X3H0]-!@[#6X3H0]@[#6X3]", 4, (0, 1, 2, 3)),
        "e12": ("[#6X3]@[#6X3H0]-!@[#6X3H0]@[#6X3]", 4, (0, 1, 2, 3)),
        "e13": (
            "[#7X2]@[#6X3]@[#6X3H0]~[#7X3][#6X3]",
            5,
            (1, 2, 3, 4),
        ),
        "e14": (
            "[#7X2]@[#6X3]@[#6X3H0]-!@[#6X2H0]#[#6X2H0]-!@[#6X3H0]@[#6X3]",
            7,
            (1, 2, 5, 6),
        ),
        "e15": ("[#8]=[#6X3][#7X3H1][#6]", 4, (0, 1, 2, 3)),
        "e16": ("[#6X3]@[#6X3H0]-!@[#6X3H0]@[#6X3]", 4, (0, 1, 2, 3)),
        "e17": (
            "[#6X3H0]@[#6X3H0]-!@[#6X2H0]#[#6X2H0]-!@[#6X3H0]@[#6X3]",
            6,
            (0, 1, 4, 5),
        ),
        "e18": (
            (
                "[#7X2]@[#6X3]@[#6X3]@[#6X3H0]-!@[#6X2H0]#[#6X2H0]-"
                "!@[#6X3H0]@[#6X3]"
            ),
            8,
            (2, 3, 6, 7),
        ),
    }

    lig_conf_data = {}
    logging.info(f"building conformer ensemble of {name}")
    generator = bbprep.generators.TorsionScanner(
        target_torsions=bbprep.generators.TargetTorsion(
            smarts=name_to_smarts[name][0],
            expected_num_atoms=name_to_smarts[name][1],
            torsion_ids=name_to_smarts[name][2],
        ),
        angle_range=range(0, 362, 10),
    )
    ensemble = generator.generate_conformers(molecule)

    if ensemble.get_num_conformers() < 2:
        raise ValueError(f"Torsions unscanned in {name}")
    logging.info(f"{ensemble} generated for {name}")

    within_dihedral = []
    for conformer in ensemble.yield_conformers():
        NCCN_dihedral = abs(calculate_NCCN_dihedral(conformer.molecule))
        if NCCN_dihedral <= dihedral_cutoff:
            within_dihedral.append(conformer)
        conformer.molecule.write(
            calc_dir / f"{name}_d_{conformer.conformer_id}.mol"
        )

    logging.info(f"{len(within_dihedral)} within dihedral_cutoff for {name}")

    rmsd_cutoff = 0.5
    num_confs = 0
    for i, conformer in enumerate(within_dihedral):
        molecule = conformer.molecule.with_centroid(np.array((0, 0, 0)))
        # Always save first one.
        if i == 0:
            first_pos_mat = molecule.get_position_matrix()
            aligned_ = molecule.clone()
        else:
            current_pos_mat = molecule.get_position_matrix()
            alignment = rmsd.kabsch(current_pos_mat, first_pos_mat)
            aligned_pos_mat = np.dot(current_pos_mat, alignment)
            aligned_ = molecule.with_position_matrix(aligned_pos_mat)

            rmsd_comparison = rmsd.rmsd(aligned_pos_mat, first_pos_mat)
            if rmsd_comparison < rmsd_cutoff:
                continue

        energy = stko.UFFEnergy().get_energy(aligned_)
        aligned_.write(calc_dir / f"{name}_s_{conformer.conformer_id}.mol")
        lig_conf_data[conformer.conformer_id] = {
            "NcentroidN_angle": calculate_N_centroid_N_angle(aligned_),
            "NCCN_dihedral": abs(calculate_NCCN_dihedral(aligned_)),
            "NN_distance": calculate_NN_distance(aligned_),
            "NN_BCN_angles": calculate_NN_BCN_angles(aligned_),
            "UFFEnergy;kj/mol": energy * 4.184,
        }
        num_confs += 1

    logging.info(
        f"{num_confs} within dihedral_cutoff and rmsd_cutoff for {name}"
    )
    with open(conf_data_file, "w") as f:
        json.dump(lig_conf_data, f)


def main():
    if not len(sys.argv) == 1:
        logging.info(f"Usage: {__file__}\n" "   Expected 0 arguments:")
        sys.exit()
    else:
        pass

    _wd = liga_path()
    _cd = calc_path()

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
        unopt_file = _wd / f"{lig}_unopt.mol"
        lowe_file = _wd / f"{lig}_lowe.mol"
        confuff_data_file = _wd / f"{lig}_conf_uff_data.json"
        systconf_data_file = _wd / f"{lig}_scan_data.json"
        unopt_mol = stk.BuildingBlock(
            smiles=lsmiles[lig],
            functional_groups=(AromaticCNCFactory(),),
        )
        unopt_mol = bbprep.FurthestFGs().modify(
            building_block=unopt_mol,
            desired_functional_groups=2,
        )
        unopt_mol.write(unopt_file)

        if not os.path.exists(confuff_data_file):
            conformer_generation_uff(
                molecule=unopt_mol,
                name=lig,
                lowe_output=lowe_file,
                conf_data_file=confuff_data_file,
                calc_dir=_cd,
            )

        if not os.path.exists(systconf_data_file):
            conformer_generation_scan(
                molecule=unopt_mol,
                name=lig,
                conf_data_file=systconf_data_file,
                dihedral_cutoff=dihedral_cutoff,
                calc_dir=_cd,
            )

    experimental_ligand_outcomes = {
        # Small, large.
        ("e11", "e10"): "yes",
        ("e12", "e10"): "yes",
        ("e11", "e14"): "yes",
        ("e12", "e14"): "yes",
        ("e11", "e13"): "yes",
        ("e12", "e13"): "yes",
        ("e16", "e17"): "yes",
        ("e16", "e10"): "yes",
        ("e13", "e14"): "no",
        ("e16", "e14"): "yes",
        ("e18", "e14"): "yes",
        ("e1", "e3"): "no",
        ("e1", "e4"): "no",
        ("e1", "e6"): "no",
        ("e3", "e2"): "no",
        ("e3", "e9"): "no",
        ("e10", "e17"): "no",
        # ("e15", "e14"): "yes",
        ("e18", "e10"): "yes",
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

    # res_file = os.path.join(_wd, "all_ligand_res.json")
    # pair_file = os.path.join(_wd, "all_pair_res.json")
    # conf_data_suffix = "conf_uff_data"
    # figure_prefix = "etkdg"

    res_file = os.path.join(_wd, "scan_ligand_res.json")
    pair_file = os.path.join(_wd, "scan_pair_res.json")
    conf_data_suffix = "scan_data"
    figure_prefix = "scan"

    structure_results = {}
    if os.path.exists(res_file):
        with open(res_file, "r") as f:
            structure_results = json.load(f)
    else:
        for ligand in ligand_smiles():
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
                continue
                if yprop != "xtb_dmsoenergy":
                    plotting.plot_vs_energy(
                        results_dict=structure_results[ligand],
                        outname=f"{figure_prefix}_ve_{ligand}_{yprop}",
                        yproperty=yprop,
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

                # Check lengths.
                # swapped_LS = False
                # try:
                #     test_N_N_lengths(
                #         large_c_dict=large_c_dict,
                #         small_c_dict=small_c_dict,
                #     )
                # except ValueError:
                #     large_c_dict = small_l_dict[small_cid]
                #     small_c_dict = large_l_dict[large_cid]
                #     swapped_LS = True

                # Test the angles are converging and diverging.
                # converging = test_converging_angles(large_c_dict)
                # diverging = test_diverging_angles(small_c_dict)

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
                )
                geom_score = abs(angle_dev - 1) + abs(length_dev - 1)

                if pair_name == "e3,e2":
                    print(cid_name, geom_score, length_dev, angle_dev)
                    print(large_c_dict, small_c_dict)
                    # input()

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

        logging.info(f"Min. geom scores for each pair:\n {min_geom_scores}")

        with open(pair_file, "w") as f:
            json.dump(pair_info, f, indent=4)

    # Get geoms core cutoff from the max, min geomscore in experimental
    # successes.
    geom_score_max = get_gs_cutoff(
        results_dict=pair_info,
        dihedral_cutoff=dihedral_cutoff,
        experimental_ligand_outcomes=experimental_ligand_outcomes,
    )
    geom_score_cutoff = round_up(geom_score_max, 2)

    logging.info(f"found a gs cutoff of: {geom_score_cutoff}")
    plotting.gs_table_plot(
        results_dict=pair_info,
        dihedral_cutoff=dihedral_cutoff,
        geom_score_cutoff=geom_score_cutoff,
        strain_cutoff=strain_cutoff,
        prefix=figure_prefix,
    )

    plotting.plot_all_ligand_pairings(
        results_dict=pair_info,
        dihedral_cutoff=dihedral_cutoff,
        geom_score_cutoff=geom_score_cutoff,
        length_score_cutoff=0.3,
        angle_score_cutoff=0.3,
        strain_cutoff=strain_cutoff,
        outname=f"{figure_prefix}_all_lp.png",
    )

    plotting.plot_all_geom_scores(
        results_dict=pair_info,
        outname=f"{figure_prefix}_all_pairs.png",
        dihedral_cutoff=dihedral_cutoff,
        strain_cutoff=strain_cutoff,
        experimental_ligand_outcomes=experimental_ligand_outcomes,
    )
    plotting.plot_all_geom_scores_mean(
        results_dict=pair_info,
        outname=f"{figure_prefix}_all_pairs_mean.png",
        dihedral_cutoff=dihedral_cutoff,
        strain_cutoff=strain_cutoff,
        experimental_ligand_outcomes=experimental_ligand_outcomes,
    )

    plotting.plot_geom_scores_vs_threshold(
        results_dict=pair_info,
        dihedral_cutoff=dihedral_cutoff,
        outname=f"{figure_prefix}_gs_cutoff.png",
        experimental_ligand_outcomes=experimental_ligand_outcomes,
    )

    plotting.gs_table(
        results_dict=pair_info,
        dihedral_cutoff=dihedral_cutoff,
        geom_score_cutoff=geom_score_cutoff,
        strain_cutoff=strain_cutoff,
    )

    plotting.plot_conformer_props(
        structure_results=structure_results,
        outname=f"{figure_prefix}_conformer_properties.png",
        dihedral_cutoff=dihedral_cutoff,
        strain_cutoff=strain_cutoff,
        experimental_ligand_outcomes=experimental_ligand_outcomes,
        low_energy_values=low_energy_values,
    )

    plotting.plot_all_geom_scores_density(
        results_dict=pair_info,
        outname=f"{figure_prefix}_all_pairs_density.png",
        dihedral_cutoff=dihedral_cutoff,
        geom_score_cutoff=geom_score_cutoff,
        experimental_ligand_outcomes=experimental_ligand_outcomes,
    )

    plotting.plot_all_geom_scores_categ(
        results_dict=pair_info,
        outname=f"{figure_prefix}_all_pairs_categorical.png",
        dihedral_cutoff=dihedral_cutoff,
        geom_score_cutoff=geom_score_cutoff,
        length_score_cutoff=geom_score_cutoff,
        angle_score_cutoff=geom_score_cutoff,
        strain_cutoff=strain_cutoff,
        experimental_ligand_outcomes=experimental_ligand_outcomes,
    )

    for pair_name in pair_info:
        small_l, large_l = pair_name.split(",")
        plotting.plot_ligand_pairing(
            results_dict=pair_info[pair_name],
            dihedral_cutoff=dihedral_cutoff,
            geom_score_cutoff=geom_score_cutoff,
            length_score_cutoff=0.3,
            angle_score_cutoff=0.3,
            strain_cutoff=strain_cutoff,
            outname=f"{figure_prefix}_lp_{small_l}_{large_l}.png",
        )

        plotting.plot_geom_scores(
            results_dict=pair_info[pair_name],
            dihedral_cutoff=dihedral_cutoff,
            geom_score_cutoff=geom_score_cutoff,
            outname=f"{figure_prefix}_gs_{small_l}_{large_l}.png",
        )

    plotting.previous_lit_table(
        results_dict=pair_info,
        dihedral_cutoff=dihedral_cutoff,
        geom_score_cutoff=geom_score_cutoff,
        strain_cutoff=strain_cutoff,
    )
    plotting.plot_analytical_data_1()
    plotting.plot_analytical_data_2()
    plotting.plot_analytical_data_3()
    raise SystemExit()

    plotting.plot_geom_scores_vs_dihedral_cutoff(
        results_dict=pair_info,
        geom_score_cutoff=geom_score_cutoff,
        outname=f"{figure_prefix}_dihedral_cutoff.png",
    )
    # plotting.plot_geom_scores_vs_max_strain(
    #     results_dict=pair_info,
    #     dihedral_cutoff=dihedral_cutoff,
    #     geom_score_cutoff=geom_score_cutoff,
    #     outname=f"{figure_prefix}_max_strain.png",
    # )

    # plotting.plot_all_geom_scores_single(
    #     results_dict=pair_info,
    #     outname=f"{figure_prefix}_all_pairs_single.png",
    #     dihedral_cutoff=dihedral_cutoff,
    #     experimental_ligand_outcomes=experimental_ligand_outcomes,
    # )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    main()
