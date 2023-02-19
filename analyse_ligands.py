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

    l_angle = (large_c_dict["NN_BCN_angles"]["NN_BCN1"] - 90) + (
        large_c_dict["NN_BCN_angles"]["NN_BCN2"] - 90
    )

    s_angle = small_c_dict["bite_angle"]
    return s_angle / l_angle


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


def main():
    if not len(sys.argv) == 1:
        logging.info(f"Usage: {__file__}\n" "   Expected 0 arguments:")
        sys.exit()
    else:
        pass

    _ld = liga_path()

    yproperties = (
        "xtb_energy",
        "NcentroidN_angle",
        "NN_distance",
        "NCCN_dihedral",
        "NN_BCN_angles",
    )

    ligand_pairings = (
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
    )

    res_file = os.path.join(_ld, "all_ligand_res.json")
    structure_results = {}
    if os.path.exists(res_file):
        with open(res_file, "r") as f:
            structure_results = json.load(f)
    else:
        for ligand in ligand_smiles():
            structure_results[ligand] = {}
            conf_data_file = _ld / f"{ligand}_conf_data.json"
            with open(conf_data_file, "r") as f:
                property_dict = json.load(f)
            for yprop in yproperties:
                plotting.plot_conf_distribution(
                    results_dict=property_dict,
                    outname=f"d_{ligand}_{yprop}",
                    yproperty=yprop,
                )

            structure_results[ligand] = property_dict

        with open(res_file, "w") as f:
            json.dump(structure_results, f)

    pair_info = {}
    min_geom_scores = {}
    for small_l, large_l in ligand_pairings:
        min_geom_score = 1e24
        pair_info[(small_l, large_l)] = {}
        small_l_dict = structure_results[small_l]
        large_l_dict = structure_results[large_l]

        # Iterate over the product of all conformers.
        for small_cid, large_cid in itertools.product(
            small_l_dict, large_l_dict
        ):

            # Calculate geom score for both sides together.
            large_c_dict = large_l_dict[large_cid]
            small_c_dict = small_l_dict[small_cid]

            # Check lengths.
            test_N_N_lengths(
                large_c_dict=large_c_dict,
                small_c_dict=small_c_dict,
            )

            # Calculate final geometrical properties.
            # T1.
            sum_angle_dev = get_angle_deviations(
                large_c_dict=large_c_dict,
                small_c_dict=small_c_dict,
            )
            # T2.
            sum_length_dev = get_ideal_length_deviation(
                large_c_dict=large_c_dict,
                small_c_dict=small_c_dict,
            )

            geom_score = sum_angle_dev / 180 + sum_length_dev / 20
            min_geom_score = min((geom_score, min_geom_score))
            pair_info[(small_l, large_l)][(small_cid, large_cid)] = {
                "geom_score": geom_score,
                "sum_length_dev": sum_length_dev,
                "sum_angle_dev": sum_angle_dev,
            }
        min_geom_scores[(small_l, large_l)] = round(min_geom_score, 2)
        plotting.plot_ligand_pairing(
            results_dict=pair_info[(small_l, large_l)],
            outname=f"lp_{small_l}_{large_l}",
        )

    logging.info(f"Min. geom scores for each pair:\n {min_geom_scores}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    main()
