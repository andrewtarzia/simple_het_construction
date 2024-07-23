#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Script to build the ligand in this project.

Author: Andrew Tarzia

"""

import logging
import sys
import json
import itertools
import numpy as np

from env_set import liga_path, calc_path
import plotting

from run_ligand_analysis import get_test_1, get_test_2
from definitions import EnvVariables


def ligand_smiles():
    return {
        # Diverging.
        "l1": "C1=NC=CC(C2=CC=C3OC4C=CC(C5C=CN=CC=5)=CC=4C3=C2)=C1",
        "l2": "C1=CC(=CC(=C1)C2=CC=NC=C2)C3=CC=NC=C3",
        "l3": "C1=CN=CC=C1C2=CC=C(S2)C3=CC=NC=C3",
        # Converging.
        "lb": (
            "C1=CN=CC2C(C3=CC=C(C#CC4N=C(C#CC5=CC=C(C6=CC=CC7C=CN=CC6="
            "7)C=C5)C=CC=4)C=C3)=CC=CC1=2"
        ),
    }


def main():
    if not len(sys.argv) == 1:
        logging.info(f"Usage: {__file__}\n" "   Expected 0 arguments:")
        sys.exit()
    else:
        pass

    _wd = liga_path()
    _cd = calc_path()

    # Small, large.
    ligand_pairings = [("l1", "lb"), ("l2", "lb"), ("l3", "lb")]
    pd_n_distances = np.arange(1.8, 2.21, 0.01)

    conf_data_suffix = "conf_uff_data"
    figure_prefix = "pdntest"

    pd_results = {}
    for pdn in pd_n_distances:
        logging.info(f"doing {pdn}...")
        structure_results = {}
        for ligand in ligand_smiles():
            structure_results[ligand] = {}
            conf_data_file = _wd / f"{ligand}_{conf_data_suffix}.json"
            with open(conf_data_file, "r") as f:
                property_dict = json.load(f)

            for cid in property_dict:
                pdi = property_dict[cid]["NN_BCN_angles"]
                # 180 - angle, to make it the angle toward the binding
                # interaction. Minus 90  to convert to the bite-angle.
                ba = ((180 - pdi["NN_BCN1"]) - 90) + ((180 - pdi["NN_BCN2"]) - 90)
                property_dict[cid]["bite_angle"] = ba

            structure_results[ligand] = property_dict

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

        pair_info = {}
        for small_l, large_l in ligand_pairings:
            pair_name = ",".join((small_l, large_l))
            pair_info[pair_name] = {}
            small_l_dict = structure_results[small_l]
            large_l_dict = structure_results[large_l]

            # Iterate over the product of all conformers.
            for small_cid, large_cid in itertools.product(small_l_dict, large_l_dict):
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
                    pdn_distance=pdn,
                )
                geom_score = abs(angle_dev - 1) + abs(length_dev - 1)

                small_energy = small_l_dict[small_cid]["UFFEnergy;kj/mol"]
                small_strain = small_energy - low_energy_values[small_l][1]
                large_energy = large_l_dict[large_cid]["UFFEnergy;kj/mol"]
                large_strain = large_energy - low_energy_values[large_l][1]
                if (
                    small_strain > EnvVariables.strain_cutoff
                    or large_strain > EnvVariables.strain_cutoff
                ):
                    continue
                # total_strain = large_strain + small_strain

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
        pd_results[pdn] = pair_info

    plotting.plot_pdntest(
        results_dict=pd_results,
        dihedral_cutoff=EnvVariables.dihedral_cutoff,
        outname=f"{figure_prefix}_test.png",
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    main()
