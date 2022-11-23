#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Script to build all cages in this project.

Author: Andrew Tarzia

"""

import logging
import sys
import glob
import os
import stk
from itertools import combinations
from dataclasses import dataclass

from topologies import M30L60, ligand_cage_topologies
from utilities import (
    AromaticCNCFactory,
    AromaticCNC,
    get_furthest_pair_FGs,
)
from env_set import cage_path, calc_path, liga_path
from optimisation import optimisation_sequence


def react_factory():
    return stk.DativeReactionFactory(
        stk.GenericReactionFactory(
            bond_orders={
                frozenset(
                    {
                        AromaticCNC,
                        stk.SingleAtom,
                    }
                ): 9,
            },
        ),
    )


def homoleptic_m2l4(metal, ligand):
    return stk.cage.M2L4Lantern(
        building_blocks={
            metal: (0, 1),
            ligand: (2, 3, 4, 5),
        },
        optimizer=stk.MCHammer(target_bond_length=2.5),
        reaction_factory=react_factory(),
    )


def homoleptic_m3l6(metal, ligand):
    return stk.cage.M3L6(
        building_blocks={
            metal: (0, 1, 2),
            ligand: range(3, 9),
        },
        optimizer=stk.MCHammer(target_bond_length=2.5),
        reaction_factory=react_factory(),
    )


def homoleptic_m4l8(metal, ligand):
    return stk.cage.M4L8(
        building_blocks={
            metal: (0, 1, 2, 3),
            ligand: range(4, 12),
        },
        optimizer=stk.MCHammer(target_bond_length=2.5),
        reaction_factory=react_factory(),
    )


def homoleptic_m6l12(metal, ligand):
    return stk.cage.M6L12Cube(
        building_blocks={
            metal: range(0, 6),
            ligand: range(6, 18),
        },
        optimizer=stk.Collapser(),
        reaction_factory=react_factory(),
    )


def homoleptic_m12l24(metal, ligand):
    return stk.cage.M12L24(
        building_blocks={
            metal: range(0, 12),
            ligand: range(12, 36),
        },
        reaction_factory=react_factory(),
    )


def homoleptic_m24l48(metal, ligand):
    return stk.cage.M24L48(
        building_blocks={
            metal: range(0, 24),
            ligand: range(24, 72),
        },
        reaction_factory=react_factory(),
    )


def homoleptic_m30l60(metal, ligand):
    return M30L60(
        building_blocks=(metal, ligand),
        reaction_factory=react_factory(),
    )


def heteroleptic_cis(metal, ligand1, ligand2):
    return stk.cage.M2L4Lantern(
        building_blocks={
            metal: (0, 1),
            ligand1: (2, 3),
            ligand2: (4, 5),
        },
        optimizer=stk.MCHammer(target_bond_length=2.5),
        reaction_factory=react_factory(),
    )


def heteroleptic_trans(metal, ligand1, ligand2):
    return stk.cage.M2L4Lantern(
        building_blocks={
            metal: (0, 1),
            ligand1: (2, 4),
            ligand2: (3, 5),
        },
        optimizer=stk.MCHammer(target_bond_length=2.5),
        reaction_factory=react_factory(),
    )


@dataclass
class CageInfo:
    tg: stk.TopologyGraph
    charge: int


def define_to_build(ligands):
    pd = stk.BuildingBlock(
        smiles="[Pd+2]",
        functional_groups=(
            stk.SingleAtom(stk.Pd(0, charge=2)) for i in range(4)
        ),
        position_matrix=[[0, 0, 0]],
    )

    lct = ligand_cage_topologies()

    to_build = {}
    for lig in ligands:
        topo_strs = lct[lig]
        if "m2" in topo_strs:
            to_build[f"m2_{lig}"] = CageInfo(
                tg=homoleptic_m2l4(pd, ligands[lig]),
                charge=4,
            )

        if "m3" in topo_strs:
            to_build[f"m3_{lig}"] = CageInfo(
                tg=homoleptic_m3l6(pd, ligands[lig]),
                charge=6,
            )

        if "m4" in topo_strs:
            to_build[f"m4_{lig}"] = CageInfo(
                tg=homoleptic_m4l8(pd, ligands[lig]),
                charge=8,
            )

        if "m6" in topo_strs:
            to_build[f"m6_{lig}"] = CageInfo(
                tg=homoleptic_m6l12(pd, ligands[lig]),
                charge=12,
            )

        if "m12" in topo_strs:
            to_build[f"m12_{lig}"] = CageInfo(
                tg=homoleptic_m12l24(pd, ligands[lig]),
                charge=24,
            )

        if "m24" in topo_strs:
            to_build[f"m24_{lig}"] = CageInfo(
                tg=homoleptic_m24l48(pd, ligands[lig]),
                charge=48,
            )

        if "m30" in topo_strs:
            to_build[f"m30_{lig}"] = CageInfo(
                tg=homoleptic_m30l60(pd, ligands[lig]),
                charge=60,
            )

    het_to_build = (
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
    for lig1, lig2 in combinations(ligands, r=2):
        l1, l2 = tuple(sorted((lig1, lig2)))
        if (l1, l2) not in het_to_build:
            continue

        het_cis_name = f"cis_{l1}_{l2}"
        to_build[het_cis_name] = CageInfo(
            tg=heteroleptic_cis(pd, ligands[l1], ligands[l2]),
            charge=4,
        )

        het_trans_name = f"trans_{l1}_{l2}"
        to_build[het_trans_name] = CageInfo(
            tg=heteroleptic_trans(pd, ligands[l1], ligands[l2]),
            charge=4,
        )

    logging.info(f"there are {len(to_build)} cages to build")
    return to_build


def main():
    if not len(sys.argv) == 1:
        logging.info(f"Usage: {__file__}\n" "   Expected 0 arguments:")
        sys.exit()
    else:
        pass

    li_path = liga_path()
    li_suffix = "_opt.mol"
    ligands = {
        i.split("/")[-1].replace(li_suffix, ""): (
            stk.BuildingBlock.init_from_file(
                path=i,
                functional_groups=(AromaticCNCFactory(),),
            )
        )
        for i in glob.glob(str(li_path / f"*{li_suffix}"))
    }
    ligands = {
        i: ligands[i].with_functional_groups(
            functional_groups=get_furthest_pair_FGs(ligands[i])
        )
        for i in ligands
    }

    _wd = cage_path()
    _cd = calc_path()

    if not os.path.exists(_wd):
        os.mkdir(_wd)

    if not os.path.exists(_cd):
        os.mkdir(_cd)

    # Define cages to build.
    cages_to_build = define_to_build(ligands)
    # Build them all.
    for cage_name in cages_to_build:
        cage_info = cages_to_build[cage_name]
        unopt_file = os.path.join(_wd, f"{cage_name}_unopt.mol")
        opt_file = os.path.join(_wd, f"{cage_name}_opt.mol")

        logging.info(f"building {cage_name}")
        unopt_mol = stk.ConstructedMolecule(cage_info.tg)
        unopt_mol.write(unopt_file)

        if not os.path.exists(opt_file):
            logging.info(f"optimising {cage_name}")
            opt_mol = optimisation_sequence(
                mol=unopt_mol,
                name=cage_name,
                charge=cage_info.charge,
                calc_dir=_cd,
            )
            opt_mol.write(opt_file)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    main()
