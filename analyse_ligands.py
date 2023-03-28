#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Script to analyse all ligands constructed.

Author: Andrew Tarzia

"""

import logging
import sys
import numpy as np
import itertools
import os
import json

from env_set import liga_path
from build_ligands import ligand_smiles
import plotting


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
    l_angle_1 = 180 - large_c_dict["NN_BCN_angles"]["NN_BCN1"]
    l_angle_2 = 180 - large_c_dict["NN_BCN_angles"]["NN_BCN2"]

    s_angle_1 = 180 - small_c_dict["NN_BCN_angles"]["NN_BCN1"]
    s_angle_2 = 180 - small_c_dict["NN_BCN_angles"]["NN_BCN2"]

    interior_angles = l_angle_1 + l_angle_2 + s_angle_1 + s_angle_2

    return interior_angles / 360


def get_test_2(large_c_dict, small_c_dict):

    sNN_dist = small_c_dict["NN_distance"]
    lNN_dist = large_c_dict["NN_distance"]
    # Minus 180 to become internal angle of trapezoid.
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


def test_converging_angles(large_c_dict):
    # Minus 180 to become internal angle of trapezoid.
    l_angle1 = 180 - large_c_dict["NN_BCN_angles"]["NN_BCN1"]
    l_angle2 = 180 - large_c_dict["NN_BCN_angles"]["NN_BCN2"]
    if l_angle1 > 90 or l_angle2 > 90:
        return False
    return True


def test_diverging_angles(small_c_dict):
    # Minus 180 to become internal angle of trapezoid.
    s_angle1 = 180 - small_c_dict["NN_BCN_angles"]["NN_BCN1"]
    s_angle2 = 180 - small_c_dict["NN_BCN_angles"]["NN_BCN2"]
    if s_angle1 < 90 or s_angle2 < 90:
        return False
    return True


def main():
    if not len(sys.argv) == 1:
        logging.info(f"Usage: {__file__}\n" "   Expected 0 arguments:")
        sys.exit()
    else:
        pass

    _ld = liga_path()

    yproperties = (
        # "xtb_dmsoenergy",
        "NcentroidN_angle",
        "NN_distance",
        "NCCN_dihedral",
        "NN_BCN_angles",
        "bite_angle",
    )

    # experimental_ligands = (
    #     "e1",
    #     "e2",
    #     "e3",
    #     "e4",
    #     "e5",
    #     "e6",
    #     "e7",
    #     "e8",
    #     "e9",
    #     "e10",
    #     "e11",
    #     "e12",
    #     "e13",
    #     "e14",
    #     "e15",
    #     "e16",
    #     "e17",
    # )
    experimental_ligand_outcomes = {
        # Small, large.
        ("e1", "e3"): "no",
        ("e3", "e2"): "no",
        ("e1", "e4"): "no",
        ("e1", "e6"): "no",
        ("e11", "e10"): "yes",
        ("e13", "e14"): "no",
        ("e11", "e14"): "yes",
        ("e12", "e14"): "yes",
        ("e11", "e13"): "yes",
        ("e12", "e13"): "yes",
        ("e15", "e14"): "yes",
        ("e16", "e17"): "yes",
        ("e16", "e10"): "yes",
        ("e10", "e17"): "no",
        ("e18", "e10"): "yes",
        ("e16", "e14"): "yes",
        ("e18", "e14"): "yes",
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

    res_file = os.path.join(_ld, "all_ligand_res.json")
    structure_results = {}
    if os.path.exists(res_file):
        with open(res_file, "r") as f:
            structure_results = json.load(f)
    else:
        for ligand in ligand_smiles():
            structure_results[ligand] = {}
            conf_data_file = _ld / f"{ligand}_conf_uff_data.json"
            with open(conf_data_file, "r") as f:
                property_dict = json.load(f)

            for cid in property_dict:
                pdi = property_dict[cid]["NN_BCN_angles"]
                ba = (90 - pdi["NN_BCN1"]) + (90 - pdi["NN_BCN2"])
                property_dict[cid]["bite_angle"] = ba

            structure_results[ligand] = property_dict

            for yprop in yproperties:
                plotting.plot_single_distribution(
                    results_dict=structure_results[ligand],
                    outname=f"d_{ligand}_{yprop}",
                    yproperty=yprop,
                )
                continue
                if yprop != "xtb_dmsoenergy":
                    plotting.plot_vs_energy(
                        results_dict=structure_results[ligand],
                        outname=f"ve_{ligand}_{yprop}",
                        yproperty=yprop,
                    )

        with open(res_file, "w") as f:
            json.dump(structure_results, f, indent=4)

    pair_file = os.path.join(_ld, "all_pair_res.json")
    if os.path.exists(pair_file):
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
                swapped_LS = False
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
                converging = test_converging_angles(large_c_dict)
                diverging = test_diverging_angles(small_c_dict)

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

                min_geom_score = min((geom_score, min_geom_score))
                pair_info[pair_name][cid_name] = {
                    "geom_score": geom_score,
                    "swapped_LS": swapped_LS,
                    "converging": converging,
                    "diverging": diverging,
                    "large_dihedral": large_c_dict["NCCN_dihedral"],
                    "small_dihedral": small_c_dict["NCCN_dihedral"],
                    "angle_deviation": angle_dev,
                    "length_deviation": length_dev,
                    "small_NCN_angle": small_c_dict["NcentroidN_angle"],
                    "large_NCN_angle": large_c_dict["NcentroidN_angle"],
                    # "small_energy": small_l_dict[small_cid][
                    #     "xtb_dmsoenergy"
                    # ],
                    # "large_energy": large_l_dict[large_cid][
                    #     "xtb_dmsoenergy"
                    # ],
                }
            min_geom_scores[pair_name] = round(min_geom_score, 2)

        logging.info(
            f"Min. geom scores for each pair:\n {min_geom_scores}"
        )
        with open(pair_file, "w") as f:
            json.dump(pair_info, f, indent=4)

    dihedral_cutoff = 10
    plotting.plot_all_geom_scores_categorigcal(
        results_dict=pair_info,
        outname="all_pairs_categorical",
        dihedral_cutoff=dihedral_cutoff,
        experimental_ligand_outcomes=experimental_ligand_outcomes,
    )
    plotting.plot_geom_scores_vs_threshold(
        results_dict=pair_info,
        dihedral_cutoff=dihedral_cutoff,
        outname="gs_cutoff",
    )

    plotting.plot_all_geom_scores_density(
        results_dict=pair_info,
        outname="all_pairs_density",
        dihedral_cutoff=dihedral_cutoff,
        experimental_ligand_outcomes=experimental_ligand_outcomes,
    )
    plotting.plot_all_geom_scores_single(
        results_dict=pair_info,
        outname="all_pairs_single",
        dihedral_cutoff=dihedral_cutoff,
        experimental_ligand_outcomes=experimental_ligand_outcomes,
    )
    plotting.plot_all_geom_scores(
        results_dict=pair_info,
        outname="all_pairs",
        dihedral_cutoff=dihedral_cutoff,
        experimental_ligand_outcomes=experimental_ligand_outcomes,
    )

    plotting.plot_geom_scores_vs_dihedral_cutoff(
        results_dict=pair_info,
        outname="dihedral_cutoff",
    )

    for pair_name in pair_info:
        small_l, large_l = pair_name.split(",")
        plotting.plot_ligand_pairing(
            results_dict=pair_info[pair_name],
            max_dihedral=dihedral_cutoff,
            outname=f"lp_{small_l}_{large_l}",
        )
        plotting.plot_geom_scores(
            results_dict=pair_info[pair_name],
            max_dihedral=dihedral_cutoff,
            outname=f"gs_{small_l}_{large_l}",
        )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    main()
