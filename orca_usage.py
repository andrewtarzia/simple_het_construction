#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Script to write input and analyse for Orca 4.2 DFT calculations.

Author: Andrew Tarzia

"""

import os
import sys
import stk
import stko
import logging
import json

from env_set import (
    cage_path,
    liga_path,
    dft_path,
)


def get_orca_energy(
    filename,
    front_splitter='FINAL SINGLE POINT ENERGY',
    back_splitter='-------------------------',
):

    with open(filename, 'r') as f:
        data = f.read()

    energy = data.split(front_splitter)
    energy = energy[-1].split(back_splitter)[0]
    return float(energy)  # a.u.


def collate_orca_energies(file_list):

    collated_energies = {}
    for file in file_list:
        splits = file.split('/')
        mol_name = splits[1]
        splits = mol_name.split('_')
        mol_name = splits[1].lower()+'_'+splits[2][0].upper()
        energy = get_orca_energy(file)
        collated_energies[mol_name] = energy

    return collated_energies


def write_top_section(basename, method, solvent):

    if solvent is None:
        solvent_s = ''
    else:
        solvent_s = solvent

    if method == 'sp' or method == 'sp_opt':
        string = (
            f'! DFT SP B97-3c TightSCF printbasis {solvent_s} Grid6 '
            'NOFINALGRID SlowConv '
            '\n\n'
            f'%base "{basename}"\n'
            '%maxcore 3000\n'
            '%scf\n   MaxIter 2000\nend\n'
        )
    elif method == 'opt':
        string = (
            f'! DFT COPT B97-3c TightSCF printbasis {solvent_s} Grid6 '
            'NOFINALGRID SlowConv '
            '\n\n'
            f'%base "{basename}"\n'
            '%maxcore 3000\n'
            '%scf\n   MaxIter 2000\nend\n'
        )

    return string


def write_proc_section(np):

    return f'%pal\n   nprocs {np}\nend\n\n'


def write_molecule_section(filename, charge):

    multi = 1

    return f'* xyzfile {charge} {multi} {filename}\n'


def write_input_file(
    prefix,
    xyz_file,
    np,
    directory,
    method,
    charge,
    solvent,
):

    basename = f'_{prefix}_{method}'
    infile = f'{prefix}_{method}.in'

    string = write_top_section(
        basename=basename,
        method=method,
        solvent=solvent,
    )
    string += write_proc_section(np)
    string += write_molecule_section(xyz_file, charge)

    with open(os.path.join(directory, f'{infile}'), 'w') as f:
        f.write(string)


def setup_calculations(structures_to_run, methods, directory):
    num_proc = 32
    for name, s_file, charge in structures_to_run:
        struct = stk.BuildingBlock.init_from_file(str(s_file))
        for method in methods:
            prefix = f'{method}_{name}'
            if method == 'sp_opt':
                xyz_file = f'_opt_{name}_opt.xyz'
            else:
                struct.write(os.path.join(directory, f'{prefix}.xyz'))
                xyz_file = f'{prefix}.xyz'

            write_input_file(
                prefix=prefix,
                xyz_file=xyz_file,
                np=num_proc,
                directory=directory,
                method=method,
                charge=charge,
                solvent=None,
            )


def get_orca_energy(
    filename,
    front_splitter='FINAL SINGLE POINT ENERGY',
    back_splitter='-------------------------',
):

    with open(filename, 'r') as f:
        data = f.read()

    energy = data.split(front_splitter)
    energy = energy[-1].split(back_splitter)[0]
    return float(energy)  # a.u.


def analyse_calculations(structures_to_run, methods, directory):

    dft_data = {}
    for name, s_file, charge in structures_to_run:
        struct = stk.BuildingBlock.init_from_file(str(s_file))
        for method in methods:
            prefix = f'{method}_{name}'
            if method == 'opt':
                xyz_file = directory / f'_opt_{name}_opt.xyz'
                out_mol_file = directory / f'opt_{name}.mol'
                opt_struct = struct.with_structure_from_file(xyz_file)
                opt_struct.write(out_mol_file)
                # Get RMSD between DFT and XTB.
                rmsd = (
                    stko.RmsdCalculator(
                        struct
                    ).get_results(opt_struct).get_rmsd()
                )
                logging.info(f'RMSD {name}: {round(rmsd, 2)}A')
            else:
                rmsd = None

            out_file = directory / f'{prefix}_{method}.out'
            if not os.path.exists(out_file):
                raise FileNotFoundError(f'{out_file} not found.')

            # Get cage energy.
            energy = get_orca_energy(
                filename=out_file,
            )
            logging.info(f'energy {prefix}: {round(energy, 8)} a.u.')
            dft_data[prefix] = {
                'energy': energy,
                'rmsd': rmsd
            }

    with open(directory / 'dft_output.json', 'w') as f:
        json.dump(dft_data, f, indent=4)


def main():
    if (not len(sys.argv) == 2):
        logging.info(
            """
            Usage: orca_usage.py stage
                stage (str) - must be "setup" or "analyse"

            """
        )
        sys.exit()
    else:
        state = sys.argv[1]
        if state not in ('setup', 'analyse'):
            raise ValueError('stage  must be "setup" or "analyse"')

    li_path = liga_path()
    ca_path = cage_path()
    _wd = dft_path()

    if not os.path.exists(_wd):
        os.mkdir(_wd)

    methods = ('opt', 'sp', 'sp_opt')
    structures_to_run = (
        # name, structure file, charge
        ('m2_l1', ca_path / 'm2_l1_opt.mol', 4),
        ('m2_l2', ca_path / 'm2_l2_opt.mol', 4),
        ('m2_l3', ca_path / 'm2_l3_opt.mol', 4),
        ('m2_la', ca_path / 'm2_la_opt.mol', 4),
        ('m2_lb', ca_path / 'm2_lb_opt.mol', 4),
        ('m2_lc', ca_path / 'm2_lc_opt.mol', 4),
        ('m2_ld', ca_path / 'm2_ld_opt.mol', 4),

        ('m3_l1', ca_path / 'm2_l1_opt.mol', 6),
        ('m3_l2', ca_path / 'm2_l2_opt.mol', 6),
        ('m3_l3', ca_path / 'm2_l3_opt.mol', 6),
        ('m3_la', ca_path / 'm2_la_opt.mol', 6),
        ('m3_lb', ca_path / 'm2_lb_opt.mol', 6),
        ('m3_lc', ca_path / 'm2_lc_opt.mol', 6),
        ('m3_ld', ca_path / 'm2_ld_opt.mol', 6),

        ('m4_l1', ca_path / 'm2_l1_opt.mol', 8),
        ('m4_l2', ca_path / 'm2_l2_opt.mol', 8),
        ('m4_l3', ca_path / 'm2_l3_opt.mol', 8),
        ('m4_la', ca_path / 'm2_la_opt.mol', 8),
        ('m4_lb', ca_path / 'm2_lb_opt.mol', 8),
        ('m4_lc', ca_path / 'm2_lc_opt.mol', 8),
        ('m4_ld', ca_path / 'm2_ld_opt.mol', 8),

        ('cis_l1_la', ca_path / 'cis_l1_la_opt.mol', 4),
        ('cis_l1_lb', ca_path / 'cis_l1_lb_opt.mol', 4),
        ('cis_l1_lc', ca_path / 'cis_l1_lc_opt.mol', 4),
        ('cis_l1_ld', ca_path / 'cis_l1_ld_opt.mol', 4),
        ('trans_l1_la', ca_path / 'trans_l1_la_opt.mol', 4),
        ('trans_l1_lb', ca_path / 'trans_l1_lb_opt.mol', 4),
        ('trans_l1_lc', ca_path / 'trans_l1_lc_opt.mol', 4),
        ('trans_l1_ld', ca_path / 'trans_l1_ld_opt.mol', 4),

        ('cis_l2_la', ca_path / 'cis_l2_la_opt.mol', 4),
        ('cis_l2_lb', ca_path / 'cis_l2_lb_opt.mol', 4),
        ('cis_l2_lc', ca_path / 'cis_l2_lc_opt.mol', 4),
        ('cis_l2_ld', ca_path / 'cis_l2_ld_opt.mol', 4),
        ('trans_l2_la', ca_path / 'trans_l2_la_opt.mol', 4),
        ('trans_l2_lb', ca_path / 'trans_l2_lb_opt.mol', 4),
        ('trans_l2_lc', ca_path / 'trans_l2_lc_opt.mol', 4),
        ('trans_l2_ld', ca_path / 'trans_l2_ld_opt.mol', 4),

        ('cis_l3_la', ca_path / 'cis_l3_la_opt.mol', 4),
        ('cis_l3_lb', ca_path / 'cis_l3_lb_opt.mol', 4),
        ('cis_l3_lc', ca_path / 'cis_l3_lc_opt.mol', 4),
        ('cis_l3_ld', ca_path / 'cis_l3_ld_opt.mol', 4),
        ('trans_l3_la', ca_path / 'trans_l3_la_opt.mol', 4),
        ('trans_l3_lb', ca_path / 'trans_l3_lb_opt.mol', 4),
        ('trans_l3_lc', ca_path / 'trans_l3_lc_opt.mol', 4),
        ('trans_l3_ld', ca_path / 'trans_l3_ld_opt.mol', 4),
    )

    if state == 'setup':
        setup_calculations(structures_to_run, methods, _wd)
    elif state == 'analyse':
        analyse_calculations(structures_to_run, methods, _wd)


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s | %(levelname)s | %(message)s',
    )
    main()
