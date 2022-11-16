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
import stko
from rdkit.Chem import AllChem as rdkit
from rdkit.Chem import Draw

from env_set import liga_path, calc_path, xtb_path, figu_path
from utilities import (
    AromaticCNCFactory,
    update_from_rdkit_conf,
    calculate_N_centroid_N_angle,
    get_furthest_pair_FGs,
    get_xtb_energy,
)


def draw_grid(names, smiles, image_file):
    mols = [rdkit.MolFromSmiles(i) for i in smiles]
    svg = Draw.MolsToGridImage(
        mols,
        molsPerRow=3,
        subImgSize=(300, 100),
        legends=names,
        useSVG=True,
    )
    with open(image_file, "w") as f:
        f.write(svg)


def select_conformer(molecule, name, lowe_output, calc_dir):
    """
    Select and optimize a conformer with desired directionality.

    Currently:
        Best directionality will be defined by the smallest
        N-ligand centroid-N angle.

    """

    confs = molecule.to_rdkit_mol()
    etkdg = rdkit.srETKDGv3()
    etkdg.randomSeed = 1000
    cids = rdkit.EmbedMultipleConfs(
        mol=confs,
        numConfs=100,
        params=etkdg,
    )

    min_angle = 10000
    min_energy = 1e24
    for cid in cids:
        # Update stk_mol to conformer geometry.
        new_mol = update_from_rdkit_conf(
            stk_mol=molecule, rdk_mol=confs, conf_id=cid
        )
        # Need to define the functional groups.
        new_mol = stk.BuildingBlock.init_from_molecule(
            molecule=new_mol, functional_groups=[AromaticCNCFactory()]
        )
        # Only get two FGs.
        new_mol = new_mol.with_functional_groups(
            functional_groups=get_furthest_pair_FGs(new_mol)
        )
        logging.info(f"xtb opt of {name} conformer {cid}")
        xtb_opt = stko.XTB(
            xtb_path=xtb_path(),
            output_dir=calc_dir / f"{name}_{cid}_ligxtb",
            gfn_version=2,
            num_cores=6,
            opt_level="normal",
            max_runs=1,
            calculate_hessian=False,
            unlimited_memory=True,
        )
        new_mol = xtb_opt.optimize(mol=new_mol)
        angle = calculate_N_centroid_N_angle(new_mol)
        charge = 0
        cid_name = f"{name}_{cid}_ligey"
        energy = get_xtb_energy(new_mol, cid_name, charge, calc_dir)
        if angle < min_angle:
            logging.info(f"new selected conformer: {cid}")
            min_angle = angle
            final_molecule = stk.BuildingBlock.init_from_molecule(
                new_mol,
            )

        if energy < min_energy:
            logging.info(f"new lowest energy conformer: {cid}")
            min_energy = energy
            new_mol.write(str(lowe_output))
            with open(lowe_output.replace(".mol", "_xtb.ey"), "w") as f:
                f.write(f"{min_energy}\n")

    return final_molecule


def main():
    if not len(sys.argv) == 1:
        logging.info(f"Usage: {__file__}\n" "   Expected 0 arguments:")
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
        "l1": "C1=NC=CC(C2=CC=C3OC4C=CC(C5C=CN=CC=5)=CC=4C3=C2)=C1",
        "l2": "C1=CC(=CC(=C1)C2=CC=NC=C2)C3=CC=NC=C3",
        "l3": "C1=CN=CC=C1C2=CC=C(S2)C3=CC=NC=C3",
        # Converging.
        "la": (
            "C1=CN=CC2C(C3=CC=C(C#CC4=CC5C6C=C(C#CC7=CC=C(C8=CC=CC9C=C"
            "N=CC8=9)C=C7)C=CC=6OC=5C=C4)C=C3)=CC=CC1=2"
        ),
        "lb": (
            "C1=CN=CC2C(C3=CC=C(C#CC4N=C(C#CC5=CC=C(C6=CC=CC7C=CN=CC6="
            "7)C=C5)C=CC=4)C=C3)=CC=CC1=2"
        ),
        "lc": (
            "C1C2=C(C(=CC=C2)C2C=CC(C#CC3=CC=CC(C#CC4C=CC(C5C6=C(C=CN="
            "C6)C=CC=5)=CC=4)=C3)=CC=2)C=NC=1"
        ),
        "ld": (
            "C1C2=C(C(=CC=C2)C2C=CC(C#CC3=CC=C(C#CC4C=CC(C5C6=C(C=CN=C"
            "6)C=CC=5)=CC=4)S3)=CC=2)C=NC=1"
        ),
    }

    for lig in ligand_smiles:
        unopt_file = _wd / f"{lig}_unopt.mol"
        opt_file = _wd / f"{lig}_opt.mol"
        lowe_file = _wd / f"{lig}_lowe.mol"
        unopt_mol = stk.BuildingBlock(
            smiles=ligand_smiles[lig],
            functional_groups=(AromaticCNCFactory(),),
        )
        unopt_mol.write(unopt_file)

        if not os.path.exists(opt_file):
            logging.info(f"selecting construction ligand for {lig}")
            opt_mol = select_conformer(
                molecule=unopt_mol,
                name=lig,
                lowe_output=lowe_file,
                calc_dir=_cd,
            )
            opt_mol.write(opt_file)

    draw_grid(
        names=[i for i in ligand_smiles],
        smiles=[ligand_smiles[i] for i in ligand_smiles],
        image_file=str(figu_path() / "ligands.svg"),
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    main()
