#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Script to analyse all ligands constructed.

Author: Andrew Tarzia

"""

import logging
import glob
import sys
import os
import json
import stk

from env_set import cage_path, calc_path, liga_path  # dft_path,
from utilities import (
    get_order_values,
    get_xtb_energy,
    AromaticCNCFactory,
    get_furthest_pair_FGs,
    get_xtb_strain,
    # get_dft_opt_energy,
    # get_dft_preopt_energy,
    # get_dft_strain,
)
from build_ligands import ligand_smiles
from pywindow_module import PyWindow
from inflation import PoreMapper
import plotting
from topologies import ligand_cage_topologies


def get_min_order_parameter(molecule):

    order_results = get_order_values(mol=molecule, metal=46)
    return order_results["sq_plan"]["min"]


def main():
    if not len(sys.argv) == 1:
        logging.info(f"Usage: {__file__}\n" "   Expected 0 arguments:")
        sys.exit()
    else:
        pass

    _cd = calc_path()
    _ld = liga_path()

    res_file = os.path.join(_ld, "all_ligand_res.json")
    if os.path.exists(res_file):
        with open(res_file, "r") as f:
            structure_results = json.load(f)
    else:
        for ligand in ligand_smiles():
            print(ligand)
            conf_data_file = _ld / f"{ligand}_conf_data.json"
            with open(conf_data_file, "r") as f:
                property_dict = json.load(f)
            print(property_dict)
            raise SystemExit()
            name = s_file.split("/")[-1].replace("_opt.mol", "")
            splits = name.split("_")
            if len(splits) == 2:
                prefix, lname = splits
                if prefix not in ligand_cage_topologies()[lname]:
                    continue
            else:
                prefix = splits[0]
                lname = None

            properties = property_dictionary[prefix]
            charge = properties["charge"]
            exp_lig = properties["exp_lig"]
            molecule = stk.BuildingBlock.init_from_file(s_file)

            structure_results[name]["xtb_energy"] = get_xtb_energy(
                molecule=molecule,
                name=name,
                charge=charge,
                calc_dir=_cd,
            )
            # structure_results[name][
            #     "dft_preopt_energy"
            # ] = get_dft_preopt_energy(molecule, name, dft_directory)
            # structure_results[name][
            #     "dft_opt_energy"
            # ] = get_dft_opt_energy(molecule, name, dft_directory)
            logging.info("not doing DFT stuff yet.")

            structure_results[name]["xtb_lig_strain"] = get_xtb_strain(
                molecule=molecule,
                name=name,
                liga_dir=_ld,
                calc_dir=_cd,
                exp_lig=exp_lig,
            )
            # structure_results[name]["dft_lig_strain"] = get_dft_strain(
            #     molecule, name, charge, _cd
            # )

            min_order_param = get_min_order_parameter(molecule)
            structure_results[name]["min_order_param"] = min_order_param

            structure_results[name]["pw_results"] = PyWindow(
                name, _cd
            ).get_results(molecule)
            structure_results[name]["pm_results"] = PoreMapper(
                name, _cd
            ).get_results(molecule)

        with open(res_file, "w") as f:
            json.dump(structure_results, f)

    raise SystemExit()
    plotting.plot_property(
        results_dict=structure_results,
        outname="cage_ops",
        yproperty="min_order_param",
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    main()
