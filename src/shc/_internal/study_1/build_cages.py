"""Script to build all cages in this project."""

import glob
import logging
from dataclasses import dataclass
from itertools import combinations

import stk

from shc.definitions import Study1EnvVariables

from .optimisation import optimisation_sequence
from .topologies import (
    M30L60,
    heteroleptic_cages,
    ligand_cage_topologies,
)
from .utilities import AromaticCNC, AromaticCNCFactory, get_furthest_pair_FGs

react_factory = stk.DativeReactionFactory(
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


def homoleptic_m2l4(
    metal: stk.BuildingBlock,
    ligand: stk.BuildingBlock,
) -> stk.ConstructedMolecule:
    """Make cage."""
    return stk.cage.M2L4Lantern(
        building_blocks={
            metal: (0, 1),
            ligand: (2, 3, 4, 5),
        },
        optimizer=stk.MCHammer(target_bond_length=2.5),
        reaction_factory=react_factory,
    )


def homoleptic_m3l6(
    metal: stk.BuildingBlock,
    ligand: stk.BuildingBlock,
) -> stk.ConstructedMolecule:
    """Make cage."""
    return stk.cage.M3L6(
        building_blocks={
            metal: (0, 1, 2),
            ligand: range(3, 9),
        },
        optimizer=stk.MCHammer(target_bond_length=2.5),
        reaction_factory=react_factory,
    )


def homoleptic_m4l8(
    metal: stk.BuildingBlock,
    ligand: stk.BuildingBlock,
) -> stk.ConstructedMolecule:
    """Make cage."""
    return stk.cage.M4L8(
        building_blocks={
            metal: (0, 1, 2, 3),
            ligand: range(4, 12),
        },
        optimizer=stk.MCHammer(target_bond_length=2.5),
        reaction_factory=react_factory,
    )


def homoleptic_m6l12(
    metal: stk.BuildingBlock,
    ligand: stk.BuildingBlock,
) -> stk.ConstructedMolecule:
    """Make cage."""
    return stk.cage.M6L12Cube(
        building_blocks={
            metal: range(6),
            ligand: range(6, 18),
        },
        optimizer=stk.Collapser(),
        reaction_factory=react_factory,
    )


def homoleptic_m12l24(
    metal: stk.BuildingBlock,
    ligand: stk.BuildingBlock,
) -> stk.ConstructedMolecule:
    """Make cage."""
    return stk.cage.M12L24(
        building_blocks={
            metal: range(12),
            ligand: range(12, 36),
        },
        reaction_factory=react_factory,
    )


def homoleptic_m24l48(
    metal: stk.BuildingBlock,
    ligand: stk.BuildingBlock,
) -> stk.ConstructedMolecule:
    """Make cage."""
    return stk.cage.M24L48(
        building_blocks={
            metal: range(24),
            ligand: range(24, 72),
        },
        reaction_factory=react_factory,
    )


def homoleptic_m30l60(
    metal: stk.BuildingBlock,
    ligand: stk.BuildingBlock,
) -> stk.ConstructedMolecule:
    """Make cage."""
    return M30L60(
        building_blocks=(metal, ligand),
        reaction_factory=react_factory,
    )


def heteroleptic_cis(
    metal: stk.BuildingBlock,
    ligand1: stk.BuildingBlock,
    ligand2: stk.BuildingBlock,
) -> stk.ConstructedMolecule:
    """Make cage."""
    return stk.cage.M2L4Lantern(
        building_blocks={
            metal: (0, 1),
            ligand1: (2, 3),
            ligand2: (4, 5),
        },
        optimizer=stk.MCHammer(target_bond_length=2.5),
        reaction_factory=react_factory,
    )


def heteroleptic_trans(
    metal: stk.BuildingBlock,
    ligand1: stk.BuildingBlock,
    ligand2: stk.BuildingBlock,
) -> stk.ConstructedMolecule:
    """Make cage."""
    return stk.cage.M2L4Lantern(
        building_blocks={
            metal: (0, 1),
            ligand1: (2, 4),
            ligand2: (3, 5),
        },
        optimizer=stk.MCHammer(target_bond_length=2.5),
        reaction_factory=react_factory,
    )


@dataclass
class CageInfo:
    """Hold cage information."""

    tg: stk.TopologyGraph
    charge: int


def define_to_build(  # noqa: C901
    ligands: dict[str, stk.BuildingBlock],
) -> dict[str, CageInfo]:
    """Define cages to build."""
    pd = stk.BuildingBlock(
        smiles="[Pd+2]",
        functional_groups=(
            stk.SingleAtom(stk.Pd(0, charge=2)) for i in range(4)
        ),
        position_matrix=[[0, 0, 0]],
    )

    lct = ligand_cage_topologies()

    to_build = {}
    for lig, lvalue in ligands.items():
        try:
            topo_strs = lct[lig]
        except KeyError:
            continue

        if "m2" in topo_strs:
            to_build[f"m2_{lig}"] = CageInfo(
                tg=homoleptic_m2l4(pd, lvalue),
                charge=4,
            )

        if "m3" in topo_strs:
            to_build[f"m3_{lig}"] = CageInfo(
                tg=homoleptic_m3l6(pd, lvalue),
                charge=6,
            )

        if "m4" in topo_strs:
            to_build[f"m4_{lig}"] = CageInfo(
                tg=homoleptic_m4l8(pd, lvalue),
                charge=8,
            )

        if "m6" in topo_strs:
            to_build[f"m6_{lig}"] = CageInfo(
                tg=homoleptic_m6l12(pd, lvalue),
                charge=12,
            )

        if "m12" in topo_strs:
            to_build[f"m12_{lig}"] = CageInfo(
                tg=homoleptic_m12l24(pd, lvalue),
                charge=24,
            )

        if "m24" in topo_strs:
            to_build[f"m24_{lig}"] = CageInfo(
                tg=homoleptic_m24l48(pd, lvalue),
                charge=48,
            )

        if "m30" in topo_strs:
            to_build[f"m30_{lig}"] = CageInfo(
                tg=homoleptic_m30l60(pd, lvalue),
                charge=60,
            )

    het_to_build = heteroleptic_cages()
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

    logging.info("there are %s cages to build", len(to_build))
    return to_build


def main() -> None:
    """Run script."""
    li_path = Study1EnvVariables.liga_path
    li_suffix = "_opt.mol"
    ligands = {
        i.split("/")[-1].replace(li_suffix, ""): (
            stk.BuildingBlock.init_from_file(
                path=i,
                functional_groups=(AromaticCNCFactory(),),
            )
        )
        for i in glob.glob(str(li_path / f"*{li_suffix}"))  # noqa: PTH207
    }
    ligands = {
        i: ligands[i].with_functional_groups(
            functional_groups=get_furthest_pair_FGs(ligands[i])
        )
        for i in ligands
    }

    _wd = Study1EnvVariables.cage_path
    _cd = Study1EnvVariables.calc_path
    _wd.mkdir(exist_ok=True)
    _cd.mkdir(exist_ok=True)

    # Define cages to build.
    cages_to_build = define_to_build(ligands)
    # Build them all.
    for cage_name in cages_to_build:
        cage_info = cages_to_build[cage_name]
        unopt_file = _wd / f"{cage_name}_unopt.mol"
        opt_file = _wd / f"{cage_name}_opt.mol"

        logging.info("building %s", cage_name)
        unopt_mol = stk.ConstructedMolecule(cage_info.tg)
        unopt_mol.write(unopt_file)

        if not opt_file.exists():
            logging.info("optimising %s", cage_name)
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
