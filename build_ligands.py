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
import stk
from rdkit.Chem import AllChem as rdkit

from env_set import liga_path, calc_path
from utilities import (
    AromaticCNCFactory,
    update_from_rdkit_conf,
    calculate_N_centroid_N_angle,
)


def select_conformer(molecule):
    """
    Select and optimize a conformer with desired directionality.

    Currently:
        Best directionality will be defined by the smallest
        N-ligand centroid-N angle.

    """

    confs = molecule.to_rdkit_mol()
    etkdg = rdkit.ETKDG()
    etkdg.randomSeed = 1000
    cids = rdkit.EmbedMultipleConfs(
        mol=confs,
        numConfs=100,
        params=etkdg
    )

    min_angle = 10000
    min_cid = -10
    for cid in cids:
        # Need to define the functional groups.
        new_mol = stk.BuildingBlock.init_from_molecule(
            molecule=molecule,
            functional_groups=[AromaticCNCFactory()]
        )

        # Update stk_mol to conformer geometry.
        new_mol = update_from_rdkit_conf(
            stk_mol=new_mol,
            rdk_mol=confs,
            conf_id=cid
        )

        angle = calculate_N_centroid_N_angle(new_mol)

        if angle < min_angle:
            min_cid = cid
            min_angle = angle
            molecule = update_from_rdkit_conf(
                stk_mol=molecule,
                rdk_mol=confs,
                conf_id=min_cid
            )

    return molecule


def main():
    if (not len(sys.argv) == 1):
        logging.info(
            f'Usage: {__file__}\n'
            '   Expected 0 arguments:'
        )
        sys.exit()
    else:
        pass

    _wd = liga_path()
    _cd = calc_path()

    if not os.path.exists(_wd):
        os.mkdir(_wd)

    if not os.path.exists(_cd):
        os.mkdir(_cd)

    ligand_smiles = {
        # Diverging.
        'l1': 'C1C=C(C2C=C3C(OC4C3C=C(C3C=CN=CC=3)C=C4)=CC=2)C=CN=1',
        'l2': 'C1=CC(=CC(=C1)C2=CC=NC=C2)C3=CC=NC=C3',
        'l3': 'C1=CN=CC=C1C2=CC=C(S2)C3=CC=NC=C3',
        # Converging.
        'la': (
            'C1=CN=CC2C(C3=CC=C(C#CC4=CC5C6C=C(C#CC7=CC=C(C8=CC=CC9C=C'
            'N=CC8=9)C=C7)C=CC=6OC=5C=C4)C=C3)=CC=CC1=2'
        ),
        'lb': (
            'C1=CN=CC2C(C3=CC=C(C#CC4N=C(C#CC5=CC=C(C6=CC=CC7C=CN=CC6='
            '7)C=C5)C=CC=4)C=C3)=CC=CC1=2'
        ),
        'lc': (
            'C1C2=C(C(=CC=C2)C2C=CC(C#CC3=CC=CC(C#CC4C=CC(C5C6=C(C=CN='
            'C6)C=CC=5)=CC=4)=C3)=CC=2)C=NC=1'
        ),
        'ld': (
            'C1C2=C(C(=CC=C2)C2C=CC(C#CC3=CC=C(C#CC4C=CC(C5C6=C(C=CN=C'
            '6)C=CC=5)=CC=4)S3)=CC=2)C=NC=1'
        ),
    }

    for lig in ligand_smiles:
        unopt_file = _wd / f'{lig}_unopt.mol'
        opt_file = _wd / f'{lig}_opt.mol'
        unopt_mol = stk.BuildingBlock(
            smiles=ligand_smiles[lig],
            functional_groups=(AromaticCNCFactory(), ),
        )
        unopt_mol.write(unopt_file)

        if not os.path.exists(opt_file):
            logging.info(f'selecting construction ligand for {lig}')
            opt_mol = select_conformer(unopt_mol)
            opt_mol.write(opt_file)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s | %(levelname)s | %(message)s',
    )
    main()
