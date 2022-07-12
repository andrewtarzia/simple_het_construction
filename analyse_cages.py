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
    calculate_ligand_SE,
    get_energy,
    get_dft_opt_energy,
    get_dft_preopt_energy,
    AromaticCNCFactory,
    get_furthest_pair_FGs,
)
from pywindow_module import PyWindow
from inflation import PoreMapper
from plotting import plot_energies, plot_ops, plot_strain_energies


def get_min_order_parameter(molecule):

    order_results = get_order_values(
        mol=molecule,
        metal=46
    )
    return order_results['sq_plan']['min']


class UnexpectedNumLigands(Exception):
    ...


def get_sum_strain_energy(
    molecule,
    name,
    exp_lig,
    lowe_ligand,
    calc_dir,
):

    ls_file = os.path.join(calc_dir, f'{name}_strain.json')

    org_ligs, smiles_keys = get_organic_linkers(
        cage=molecule,
        metal_atom_nos=(46, ),
        file_prefix=f'{name}_sg',
        calc_dir=calc_dir,
    )

    num_unique_ligands = len(set(smiles_keys.values()))
    if num_unique_ligands != exp_lig:
        raise UnexpectedNumLigands(
            f'{name} had {num_unique_ligands} unique ligands'
            f', {exp_lig} were expected. Suggests bad '
            'optimization. Recommend reoptimising structure.'
        )

    lowe_ligand_energy = get_energy(
        molecule=lowe_ligand,
        name='lig',
        charge=0,
        calc_dir=calc_dir,
    )

    lse_dict = calculate_ligand_SE(
        org_ligs=org_ligs,
        lowe_ligand_energy=lowe_ligand_energy,
        output_json=f'{ls_file}.json',
        calc_dir=calc_dir,
    )

    sum_strain = sum(lse_dict.values())
    return sum_strain


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

            xtb_energy = get_energy(molecule, name, charge, _cd)
            structure_results[name]['xtb_energy'] = xtb_energy
            structure_results[name]['dft_preopt_energy'] = (
                get_dft_preopt_energy(molecule, name, dft_directory)
            )
            structure_results[name]['dft_opt_energy'] = (
                get_dft_opt_energy(molecule, name, dft_directory)
            )

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

            # sum_strain_energy = get_sum_strain_energy(
            #     molecule=molecule,
            #     name=name,
            #     exp_lig=exp_lig,
            #     lowe_ligand=lowe_ligand,
            #     calc_dir=_cd,
            # )

            # structure_results[name]['sum_strain_energy'] = (
            #     sum_strain_energy
            # )

        raise SystemExit()
        with open(structure_res_file, 'w') as f:
            json.dump(structure_results, f)

    print(structure_results)
    raise SystemExit()

    plot_energies(
        results_dict=structure_results,
        outname='cage_energies',
    )
    plot_ops(
        results_dict=structure_results,
        outname='cage_ops',
    )
    plot_strain_energies(
        results_dict=structure_results,
        outname='cage_strain_energies',
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
    )
    main()
