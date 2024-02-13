#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Script to analyse all cages constructed.

Author: Andrew Tarzia

"""

import logging
import glob
import sys
import os
import json
import stk

from env_set import cage_path, calc_path, liga_path, project_path
from utilities import (
    get_order_values,
    get_xtb_energy,
    AromaticCNCFactory,
    get_xtb_strain,
    get_furthest_pair_FGs,
    get_pore_angle,
    get_mm_distance,
)
from pywindow_module import PyWindow

import plotting
from topologies import ligand_cage_topologies, heteroleptic_cages


def get_min_order_parameter(molecule):
    order_results = get_order_values(mol=molecule, metal=46)
    return order_results["sq_plan"]["min"]


def main():
    if not len(sys.argv) == 1:
        logging.info(f"Usage: {__file__}\n" "   Expected 1 arguments:")
        sys.exit()
    else:
        pass

    li_path = liga_path()
    ligands = {
        i.split("/")[-1].replace("_opt.mol", ""): (
            stk.BuildingBlock.init_from_file(
                path=i,
                functional_groups=(AromaticCNCFactory(),),
            )
        )
        for i in glob.glob(str(li_path / "*_opt.mol"))
    }

    ligands = {
        i: ligands[i].with_functional_groups(
            functional_groups=get_furthest_pair_FGs(ligands[i])
        )
        for i in ligands
    }

    _wd = cage_path()
    _cd = calc_path()
    _ld = liga_path()
    _pd = project_path()

    property_dictionary = {
        "cis": {
            "charge": 4,
            "exp_lig": 2,
        },
        "trans": {
            "charge": 4,
            "exp_lig": 2,
        },
        "m2": {
            "charge": 4,
            "exp_lig": 1,
        },
        "m3": {
            "charge": 6,
            "exp_lig": 1,
        },
        "m4": {
            "charge": 8,
            "exp_lig": 1,
        },
        "m6": {
            "charge": 12,
            "exp_lig": 1,
        },
        "m12": {
            "charge": 24,
            "exp_lig": 1,
        },
        "m24": {
            "charge": 48,
            "exp_lig": 1,
        },
        "m30": {
            "charge": 60,
            "exp_lig": 1,
        },
    }

    structure_files = []
    hets = heteroleptic_cages()
    lct = ligand_cage_topologies()
    for ligname in lct:
        toptions = lct[ligname]
        for topt in toptions:
            sname = _wd / f"{topt}_{ligname}_opt.mol"
            structure_files.append(str(sname))

    for l1, l2 in hets:
        for topt in ("cis", "trans"):
            sname = _wd / f"{topt}_{l1}_{l2}_opt.mol"
            structure_files.append(str(sname))
    logging.info(f"there are {len(structure_files)} structures.")

    structure_results = {
        i.split("/")[-1].replace("_opt.mol", ""): {} for i in structure_files
    }
    structure_res_file = os.path.join(_wd, "all_structure_res.json")
    if os.path.exists(structure_res_file):
        with open(structure_res_file, "r") as f:
            structure_results = json.load(f)
    else:
        for s_file in structure_files:
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

            structure_results[name]["xtb_solv_opt_gasenergy_au"] = (
                get_xtb_energy(
                    molecule=molecule,
                    name=name,
                    charge=charge,
                    calc_dir=_cd,
                    solvent=None,
                )
            )
            structure_results[name]["xtb_solv_opt_dmsoenergy_au"] = (
                get_xtb_energy(
                    molecule=molecule,
                    name=name,
                    charge=charge,
                    calc_dir=_cd,
                    solvent="dmso",
                )
            )

            if prefix in ("cis", "trans", "m2"):
                structure_results[name]["pore_angle"] = get_pore_angle(
                    molecule=molecule,
                    metal_atom_num=46,
                )
                structure_results[name]["mm_distance"] = get_mm_distance(
                    molecule=molecule,
                    metal_atom_num=46,
                )

            structure_results[name]["xtb_lig_strain_au"] = get_xtb_strain(
                molecule=molecule,
                name=name,
                liga_dir=_ld,
                calc_dir=_cd,
                exp_lig=exp_lig,
                solvent="dmso",
            )

            min_order_param = get_min_order_parameter(molecule)
            structure_results[name]["min_order_param"] = min_order_param

            structure_results[name]["pw_results"] = PyWindow(
                name, _cd
            ).get_results(molecule)

        with open(structure_res_file, "w") as f:
            json.dump(structure_results, f, indent=4)

    plotting.plot_topo_energy(
        results_dict=structure_results,
        outname="main_topology_ey",
    )
    plotting.plot_topo_energy(
        results_dict=structure_results,
        outname="gas_topology_ey",
        solvent="gas",
    )
    plotting.plot_qsqp(
        results_dict=structure_results,
        outname="cage_qsqp",
        yproperty="min_order_param",
    )
    plotting.plot_property(
        results_dict=structure_results,
        outname="cage_poreangle",
        yproperty="pore_angle",
    )
    plotting.plot_property(
        results_dict=structure_results,
        outname="cage_mm_distance",
        yproperty="mm_distance",
    )
    plotting.plot_property(
        results_dict=structure_results,
        outname="cage_pw_diameter",
        yproperty="pore_diameter_opt",
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    main()
