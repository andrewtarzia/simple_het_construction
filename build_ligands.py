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
import bbprep
from rdkit.Chem import AllChem as rdkit
from rdkit.Chem import Draw

from env_set import liga_path, calc_path, xtb_path, figu_path
from utilities import (
    AromaticCNCFactory,
    update_from_rdkit_conf,
    calculate_N_centroid_N_angle,
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


def select_conformer_xtb(molecule, name, lowe_output, calc_dir):
    """
    Select and optimize a conformer with desired directionality.

    Currently:
        Best directionality will be defined by the smallest
        N-ligand centroid-N angle.

    """
    lowe_energy_output = str(lowe_output).replace(".mol", "_xtb.ey")

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
        conf_opt_file_name = str(lowe_output).replace(
            "_lowe.mol", f"_c{cid}_copt.mol"
        )
        # Update stk_mol to conformer geometry.
        new_mol = update_from_rdkit_conf(
            stk_mol=molecule, rdk_mol=confs, conf_id=cid
        )
        # Need to define the functional groups.
        new_mol = stk.BuildingBlock.init_from_molecule(
            molecule=new_mol,
            functional_groups=[AromaticCNCFactory()],
        )
        # Only get two FGs.
        new_mol = bbprep.FurthestFGs().modify(
            building_block=new_mol,
            desired_functional_groups=2,
        )
        if os.path.exists(conf_opt_file_name):
            new_mol = new_mol.with_structure_from_file(
                path=conf_opt_file_name,
            )
        else:
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
                solvent_model="alpb",
                solvent="dmso",
                solvent_grid="verytight",
            )
            new_mol = xtb_opt.optimize(mol=new_mol)
            new_mol.write(conf_opt_file_name)

        angle = calculate_N_centroid_N_angle(new_mol)
        charge = 0
        cid_name = f"{name}_{cid}_ligey"
        energy = get_xtb_energy(
            molecule=new_mol,
            name=cid_name,
            charge=charge,
            calc_dir=calc_dir,
            solvent="dmso",
        )
        if angle < min_angle:
            logging.info(f">> new selected conformer: {cid}")
            min_angle = angle
            final_molecule = stk.BuildingBlock.init_from_molecule(
                new_mol,
            )

        if energy < min_energy:
            logging.info(f">> new lowest energy conformer: {cid}")
            min_energy = energy
            new_mol.write(str(lowe_output))
            with open(lowe_energy_output, "w") as f:
                f.write(f"{min_energy}\n")

    return final_molecule


def ligand_smiles():
    return {
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
        # Experimental, but assembly tested.
        "ll1": (
            "C1=CC(C#CC2=CC3C4C=C(C#CC5=CC=CN=C5)C=CC=4N(C)C=3C=C2)=CN=" "C1"
        ),
        "ls": (
            "C(C1=CC2C3C=C(C4=CC=NC=C4)C=CC=3C(OC)=C(OC)C=2C=C1)1=CC=NC" "=C1"
        ),
        "ll2": (
            "C12C=CN=CC=1C(C#CC1=CC=C3C(C(C4=C(N3C)C=CC(C#CC3=CC=CC5C3="
            "CN=CC=5)=C4)=O)=C1)=CC=C2"
        ),
        # Experimental.
        "e1": "N1C=C(C2=CC(C3C=CC=NC=3)=CC=C2)C=CC=1",
        "e2": "C1C=CN=CC=1C#CC1C=CC=C(C#CC2C=CC=NC=2)C=1",
        "e3": "N1=CC=C(C2=CC(C3=CC=NC=C3)=CC=C2)C=C1",
        "e4": "C1=CC(C#CC2=CC(C#CC3=CC=NC=C3)=CC=C2)=CC=N1",
        "e5": "C1C=NC=CC=1C1=CC=C(C2C=CN=CC=2)S1",
        "e6": "C1N=CC=C(C2=CC=C(C3C=CN=CC=3)C=C2)C=1",
        "e7": "C1N=CC=C(C#CC2C=CC(C#CC3=CC=NC=C3)=CC=2)C=1",
        "e8": "C(OC)1=C(C2C=C(C3=CN=CC=C3OC)C=CC=2)C=NC=C1",
        "e9": (
            "C1C=C(C2C=CC(C#CC3C(C)=C(C#CC4C=CC(C5C=CN=CC=5)=CC=4)C=CC="
            "3)=CC=2)C=CN=1"
        ),
        "e10": (
            "C1=CC(C#CC2=CC3C4C=C(C#CC5=CC=CN=C5)C=CC=4N(C)C=3C=C2)=CN=" "C1"
        ),
        "e11": ("C1N=CC=CC=1C1=CC2=C(C3=C(C2(C)C)C=C(C2=CN=CC=C2)C=C3)C=C1"),
        "e12": "C1=CC=C(C2=CC3C(=O)C4C=C(C5=CN=CC=C5)C=CC=4C=3C=C2)C=N1",
        "e13": (
            "C1C=C(N2C(=O)C3=C(C=C4C(=C3)C3(C5=C(C4(C)CC3)C=C3C(C(N(C3="
            "O)C3C=CC=NC=3)=O)=C5)C)C2=O)C=NC=1"
        ),
        "e14": (
            "C1=CN=CC(C#CC2C=CC3C(=O)C4C=CC(C#CC5=CC=CN=C5)=CC=4C=3C=2)" "=C1"
        ),
        "e15": "C1CCC(C(C1)NC(=O)C2=CC=NC=C2)NC(=O)C3=CC=NC=C3",
        "e16": (
            "C(C1=CC2C3C=C(C4=CC=NC=C4)C=CC=3C(OC)=C(OC)C=2C=C1)1=CC=NC" "=C1"
        ),
        "e17": (
            "C12C=CN=CC=1C(C#CC1=CC=C3C(C(C4=C(N3C)C=CC(C#CC3=CC=CC5C3="
            "CN=CC=5)=C4)=O)=C1)=CC=C2"
        ),
        "e18": (
            "C1(=CC=NC=C1)C#CC1=CC2C3C=C(C#CC4=CC=NC=C4)C=CC=3C(OC)=C(O"
            "C)C=2C=C1"
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

    if not os.path.exists(_wd):
        os.mkdir(_wd)

    if not os.path.exists(_cd):
        os.mkdir(_cd)

    lsmiles = ligand_smiles()
    for lig in lsmiles:
        unopt_file = _wd / f"{lig}_unopt.mol"
        opt_file = _wd / f"{lig}_opt.mol"
        lowe_file = _wd / f"{lig}_lowe.mol"
        unopt_mol = stk.BuildingBlock(
            smiles=lsmiles[lig],
            functional_groups=(AromaticCNCFactory(),),
        )
        unopt_mol = bbprep.FurthestFGs().modify(
            building_block=unopt_mol,
            desired_functional_groups=2,
        )
        unopt_mol.write(unopt_file)

        if not os.path.exists(opt_file):
            logging.info(f"selecting construction ligand for {lig}")
            if lig[0] == "l":
                opt_mol = select_conformer_xtb(
                    molecule=unopt_mol,
                    name=lig,
                    lowe_output=lowe_file,
                    calc_dir=_cd,
                )
                opt_mol.write(opt_file)

    draw_grid(
        names=[i for i in lsmiles],
        smiles=[lsmiles[i] for i in lsmiles],
        image_file=str(figu_path() / "ligands.svg"),
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    main()
