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

from env_set import cage_path, calc_path, dft_path, liga_path
from utilities import (
    get_order_values,
    get_organic_linkers,
    get_xtb_energy,
    get_dft_opt_energy,
    get_dft_preopt_energy,
    AromaticCNCFactory,
    get_furthest_pair_FGs,
    get_xtb_strain,
    get_dft_strain,
)
from pywindow_module import PyWindow
from inflation import PoreMapper
import plotting


def get_min_order_parameter(molecule):

    order_results = get_order_values(
        mol=molecule,
        metal=46
    )
    return order_results['sq_plan']['min']


def main():
    if (not len(sys.argv) == 1):
        logging.info(
            f'Usage: {__file__}\n'
            '   Expected 1 arguments:'
        )
        sys.exit()
    else:
        pass

    li_path = liga_path()
    ligands = {
        i.split('/')[-1].replace('_opt.mol', ''): (
            stk.BuildingBlock.init_from_file(
                path=i,
                functional_groups=(AromaticCNCFactory(), ),
            )
        )
        for i in glob.glob(str(li_path / '*_opt.mol'))
    }
    ligands = {
        i: ligands[i].with_functional_groups(
            functional_groups=get_furthest_pair_FGs(ligands[i])
        ) for i in ligands
    }

    _wd = cage_path()
    _cd = calc_path()
    _ld = liga_path()
    dft_directory = dft_path()

    property_dictionary = {
        'cis': {
            'charge': 4,
            'exp_lig': 2,
        },
        'trans': {
            'charge': 4,
            'exp_lig': 2,
        },
        'm2': {
            'charge': 4,
            'exp_lig': 1,
        },
        'm3': {
            'charge': 6,
            'exp_lig': 1,
        },
        'm4': {
            'charge': 8,
            'exp_lig': 1,
        },
    }

    structure_files = glob.glob(os.path.join(_wd, '*_opt.mol'))
    logging.info(f'there are {len(structure_files)} structures.')
    structure_results = {
        i.split('/')[-1].replace('_opt.mol', ''): {}
        for i in structure_files
    }
    structure_res_file = os.path.join(_wd, 'all_structure_res.json')
    if os.path.exists(structure_res_file):
        with open(structure_res_file, 'r') as f:
            structure_results = json.load(f)
    else:
        for s_file in structure_files:
            name = s_file.split('/')[-1].replace('_opt.mol', '')
            prefix = name.split('_')[0]
            properties = property_dictionary[prefix]
            charge = properties['charge']
            exp_lig = properties['exp_lig']
            molecule = stk.BuildingBlock.init_from_file(s_file)

            structure_results[name]['xtb_energy'] = (
                get_xtb_energy(molecule, name, charge, _cd)
            )
            # structure_results[name]['dft_preopt_energy'] = (
            #     get_dft_preopt_energy(molecule, name, dft_directory)
            # )
            # structure_results[name]['dft_opt_energy'] = (
            #     get_dft_opt_energy(molecule, name, dft_directory)
            # )

            structure_results[name]['xtb_lig_strain'] = get_xtb_strain(
                molecule=molecule,
                name=name,
                liga_dir=_ld,
                calc_dir=_cd,
                exp_lig=exp_lig,
            )
            # structure_results[name]['dft_lig_strain'] = get_dft_strain(
            #     molecule, name, charge, _cd
            # )

            min_order_param = get_min_order_parameter(molecule)
            structure_results[name]['min_order_param'] = (
                min_order_param
            )

            structure_results[name]['pw_results'] = (
                PyWindow(name, _cd).get_results(molecule)
            )
            structure_results[name]['pm_results'] = (
                PoreMapper(name, _cd).get_results(molecule)
            )

        with open(structure_res_file, 'w') as f:
            json.dump(structure_results, f)

    plotting.plot_property(
        results_dict=structure_results,
        outname='cage_ops',
        yproperty='min_order_param',
    )
    plotting.plot_property(
        results_dict=structure_results,
        outname='cage_pw_diameter',
        yproperty='pore_diameter_opt',
    )
    plotting.plot_property(
        results_dict=structure_results,
        outname='cage_xtb_strain',
        yproperty='xtb_lig_strain',
    )
    plotting.plot_property(
        results_dict=structure_results,
        outname='het_cage_ops',
        yproperty='min_order_param',
        ignore_topos=('m3', 'm4'),
    )
    plotting.plot_property(
        results_dict=structure_results,
        outname='het_cage_pw_diameter',
        yproperty='pore_diameter_opt',
        ignore_topos=('m3', 'm4'),
    )
    plotting.plot_property(
        results_dict=structure_results,
        outname='het_cage_xtb_strain',
        yproperty='xtb_lig_strain',
        ignore_topos=('m3', 'm4'),
    )
    plotting.compare_cis_trans(
        results_dict=structure_results,
        outname='hetcf_cage_xtb_strain',
        yproperty='xtb_lig_strain',
    )
    plotting.compare_cis_trans(
        results_dict=structure_results,
        outname='hetcf_cage_ops',
        yproperty='min_order_param',
    )
    plotting.compare_cis_trans(
        results_dict=structure_results,
        outname='hetcf_cage_pw_diameter',
        yproperty='pore_diameter_opt',
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
    )
    main()
