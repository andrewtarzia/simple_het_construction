"""Module for utility functions."""

import itertools as it
import logging
import pathlib
import time

import atomlite
import bbprep
import chemiscope
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import stk
import stko

from shc.definitions import EnvVariables, res_str
from shc.matching_functions import mismatch_test
from shc.utilities import (
    merge_stk_molecules,
    remake_atomlite_molecule,
    to_atomlite_molecule,
)


def analyse_ligand_pair(  # noqa: PLR0915
    ligand1_entry: atomlite.Entry,
    ligand2_entry: atomlite.Entry,
    pair_key: str,
    pair_db_path: atomlite.Database,
) -> None:
    """Analyse a pair of ligands."""
    pair_db = atomlite.Database(pair_db_path)
    ligand1_confs = ligand1_entry.properties["conf_data"]
    ligand2_confs = ligand2_entry.properties["conf_data"]

    st = time.time()
    num_pairs = 0
    num_pairs_passed = 0
    num_failed_strain = 0
    num_failed_torsion = 0
    cid_names = []
    state1_scores = []
    state2_scores = []
    ctimes = []
    all_scores = []
    best_residual = float("inf")
    best_s1_residual = float("inf")
    best_s2_residual = float("inf")
    best_pair = None
    best_state = None
    best_s1_cid1 = None
    best_s1_cid2 = None
    best_s2_cid1 = None
    best_s2_cid2 = None
    # Iterate over the product of all conformers.
    for cid_1, cid_2 in it.product(ligand1_confs, ligand2_confs):
        cid_name = f"{cid_1}-{cid_2}"

        num_pairs += 1
        # Check strain.
        strain1 = (
            ligand1_confs[cid_1]["UFFEnergy;kj/mol"]
            - ligand1_entry.properties["min_energy;kj/mol"]
        )
        strain2 = (
            ligand2_confs[cid_2]["UFFEnergy;kj/mol"]
            - ligand2_entry.properties["min_energy;kj/mol"]
        )
        if (
            strain1 > EnvVariables.strain_cutoff
            or strain2 > EnvVariables.strain_cutoff
        ):
            num_failed_strain += 1
            continue

        # Check torsion.
        torsion1 = abs(ligand1_confs[cid_1]["NCCN_dihedral"])
        torsion2 = abs(ligand2_confs[cid_2]["NCCN_dihedral"])
        if (
            torsion1 > EnvVariables.dihedral_cutoff
            or torsion2 > EnvVariables.dihedral_cutoff
        ):
            num_failed_torsion += 1
            continue

        # Calculate geom score for both sides together.
        c_dict1 = ligand1_confs[cid_1]
        c_dict2 = ligand2_confs[cid_2]

        # Calculate final geometrical properties.
        stc = time.time()
        pair_results = mismatch_test(
            c_dict1=c_dict1,
            c_dict2=c_dict2,
            k_angle=EnvVariables.k_angle,
            k_bond=EnvVariables.k_bond,
        )
        etc = time.time()

        cid_names.append(cid_name)
        ctimes.append(float(etc - stc))
        state1_scores.append(float(pair_results.state_1_result))
        state2_scores.append(float(pair_results.state_2_result))
        all_scores.append(
            min(
                float(pair_results.state_1_result),
                float(pair_results.state_2_result),
            )
        )
        num_pairs_passed += 1
        # for idx in range(len(pair_results.state_1_parameters)):
        #     pair_db.set_property(
        # key=pair_key,# noqa: ERA001
        # path=f"{prefix_path}.state_1_parameter.{idx}",  # noqa: ERA001
        # property=float(pair_results.state_1_parameters[idx]),# noqa: ERA001
        # commit=False,# noqa: ERA001
        if (
            float(pair_results.state_1_result) < best_residual
            or float(pair_results.state_2_result) < best_residual
        ):
            best_residual = min(
                (
                    float(pair_results.state_1_result),
                    float(pair_results.state_2_result),
                )
            )
            best_pair = pair_results

            if float(pair_results.state_1_result) < float(
                pair_results.state_2_result
            ):
                best_state = 1
            elif float(pair_results.state_2_result) < float(
                pair_results.state_1_result
            ):
                best_state = 2
            else:
                # They are the same.
                pass

        if float(pair_results.state_1_result) < best_s1_residual:
            best_s1_residual = float(pair_results.state_1_result)
            best_s1_cid1 = cid_1
            best_s1_cid2 = cid_2

        if float(pair_results.state_2_result) < best_s2_residual:
            best_s2_residual = float(pair_results.state_2_result)
            best_s2_cid1 = cid_1
            best_s2_cid2 = cid_2

    ft = time.time()

    if num_pairs_passed == 0:
        logging.warning("make this smarter, need to add property entry logic")

    else:
        entry = atomlite.Entry(
            key=pair_key,
            molecule=to_atomlite_molecule(best_pair, best_state),
            properties={
                "cidnames": cid_names,
                "ctime/s": ctimes,
                "state_1_residuals": state1_scores,
                "state_2_residuals": state2_scores,
                "ligand1_key": ligand1_entry.key,
                "ligand2_key": ligand2_entry.key,
                "time_taken/s": float(ft - st),
                "num_pairs": num_pairs,
                "strain_cutoff": EnvVariables.strain_cutoff,
                "dihedral_cutoff": EnvVariables.dihedral_cutoff,
                "k_angle": EnvVariables.k_angle,
                "k_bond": EnvVariables.k_bond,
                "rmsd_threshold": EnvVariables.rmsd_threshold,
                "num_pairs_passed": num_pairs_passed,
                "num_failed_strain": num_failed_strain,
                "num_failed_torsion": num_failed_torsion,
                "mean_all": np.mean(all_scores),
                "min_all": np.min(all_scores),
                "mean_s1": np.mean(state1_scores),
                "min_s1": np.min(state1_scores),
                "mean_s2": np.mean(state1_scores),
                "min_s2": np.min(state2_scores),
                "best_s1_residual": best_s1_residual,
                "best_s1_cid1": best_s1_cid1,
                "best_s1_cid2": best_s1_cid2,
                "best_s2_residual": best_s2_residual,
                "best_s2_cid1": best_s2_cid1,
                "best_s2_cid2": best_s2_cid2,
            },
        )
        pair_db.add_entries(entry)


def reanalyse_ligand_pair(
    pair_key: str,
    pair_db_path: atomlite.Database,
) -> None:
    """Re-analyse a pair of ligands.

    This is a chance to efficiently update molecule schema or properties in a
    database to avoid rerunning everything.

    """
    pair_entry = atomlite.Database(pair_db_path).get_entry(pair_key)

    target_properties = [
        "cidnames",
        "ctime/s",
        "state_1_residuals",
        "state_2_residuals",
        "ligand1_key",
        "ligand2_key",
        "time_taken/s",
        "num_pairs",
        "strain_cutoff",
        "dihedral_cutoff",
        "k_angle",
        "k_bond",
        "rmsd_threshold",
        "num_pairs_passed",
        "num_failed_strain",
        "num_failed_torsion",
        "mean_all",
        "min_all",
        "mean_s1",
        "min_s1",
        "mean_s2",
        "min_s2",
        "best_s1_cid1",
        "best_s1_cid2",
        "best_s2_cid1",
        "best_s2_cid2",
        "best_s1_residual",
        "best_s2_residual",
    ]
    missing_properties = [
        i for i in target_properties if i not in pair_entry.properties
    ]
    if len(missing_properties) == 0:
        return

    # Check if it is missing anything needed to rerun.
    key_features = (
        "cidnames",
        "state_1_residuals",
        "state_2_residuals",
        "time_taken/s",
        "ctime/s",
        "num_pairs",
        "num_pairs_passed",
        "num_failed_strain",
        "num_failed_torsion",
    )
    if len([i for i in missing_properties if i in key_features]) > 0:
        msg = (
            f"missing a necessary feature, I deleted this entry ({pair_key})"
            f" from {pair_db_path}, please rerun"
        )
        atomlite.Database(pair_db_path).remove_entries(kets=pair_key)
        raise RuntimeError(msg)

    # Update those that are possible.
    properties = pair_entry.properties
    all_scores = []
    state1_scores = []
    state2_scores = []
    best_s1_cid1 = None
    best_s1_cid2 = None
    best_s2_cid1 = None
    best_s2_cid2 = None
    best_s1_residual = float("inf")
    best_s2_residual = float("inf")
    for cid_pair_idx, cid_name in enumerate(properties["cidnames"]):
        cid_1, cid_2 = cid_name.split("-")
        s1_score = float(properties["state_1_residuals"][cid_pair_idx])
        s2_score = float(properties["state_2_residuals"][cid_pair_idx])
        state1_scores.append(s1_score)
        state2_scores.append(s2_score)
        all_scores.append(min(s1_score, s2_score))

        if s1_score < best_s1_residual:
            best_s1_cid1 = cid_1
            best_s1_cid2 = cid_2
            best_s1_residual = s1_score

        elif s2_score < best_s2_residual:
            best_s2_cid1 = cid_1
            best_s2_cid2 = cid_2
            best_s2_residual = s2_score

    new_properties = {
        "mean_all": np.mean(all_scores),
        "min_all": np.min(all_scores),
        "mean_s1": np.mean(state1_scores),
        "min_s1": np.min(state1_scores),
        "mean_s2": np.mean(state1_scores),
        "min_s2": np.min(state2_scores),
        "best_s1_residual": best_s1_residual,
        "best_s1_cid1": best_s1_cid1,
        "best_s1_cid2": best_s1_cid2,
        "best_s2_residual": best_s2_residual,
        "best_s2_cid1": best_s2_cid1,
        "best_s2_cid2": best_s2_cid2,
    }

    atomlite.Database(pair_db_path).update_entries(
        atomlite.Entry(
            key=pair_key,
            molecule=remake_atomlite_molecule(pair_entry.molecule),
            properties=new_properties,
        )
    )


def plot_timings(
    pair_db_path: pathlib.Path,
    figures_dir: pathlib.Path,
    filename: str,
) -> None:
    """Make plot."""
    fig, (ax, ax1) = plt.subplots(ncols=2, figsize=(16, 5))

    dataframe = atomlite.Database(pair_db_path).get_property_df(
        properties=[
            "$.time_taken/s",
            "$.num_pairs",
            "$.num_pairs_passed",
            "$.ctime/s",
        ]
    )

    ax.scatter(
        dataframe["$.num_pairs_passed"],
        dataframe["$.time_taken/s"],
        c="tab:blue",
        s=50,
        ec="k",
        rasterized=True,
    )
    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel("num. pairs passed", fontsize=16)
    ax.set_ylabel("time [s]", fontsize=16)
    ax.set_xscale("log")
    ax.set_yscale("log")

    xwidth = 0.05
    xbins = np.arange(0 - xwidth, 1 + xwidth, xwidth)
    ax1.hist(
        x=dataframe["$.ctime/s"].list.explode(),
        bins=xbins,
        density=True,
        histtype="stepfilled",
        stacked=True,
        linewidth=1.0,
        alpha=1.0,
        edgecolor="k",
        facecolor="tab:blue",
    )

    ax1.tick_params(axis="both", which="major", labelsize=16)
    ax1.set_xlabel("calculation time [s]", fontsize=16)

    fig.tight_layout()
    fig.savefig(
        figures_dir / f"{filename}.png",
        dpi=360,
        bbox_inches="tight",
    )
    plt.close()


def plot_states(
    pair_db_path: pathlib.Path,
    figures_dir: pathlib.Path,
    filename: str,
) -> None:
    """Make plot."""
    fig, ax = plt.subplots(ncols=1, figsize=(5, 5))

    dataframe = atomlite.Database(pair_db_path).get_property_df(
        properties=["$.mean_s1", "$.min_s1", "$.mean_s2", "$.min_s2"]
    )

    ax.scatter(
        dataframe["$.mean_s1"],
        dataframe["$.mean_s2"],
        c="tab:blue",
        s=30,
        ec="k",
        label="mean",
        rasterized=True,
        zorder=2,
    )

    ax.scatter(
        dataframe["$.min_s1"],
        dataframe["$.min_s2"],
        c="tab:orange",
        s=20,
        ec="k",
        label="min",
        rasterized=True,
        zorder=1,
    )
    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel(f"1 {res_str}", fontsize=16)
    ax.set_ylabel(f"2 {res_str}", fontsize=16)
    ax.plot((0, 50), (0, 50), c="k", zorder=-1)
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 50)
    ax.legend(fontsize=16)
    ax.axvspan(
        xmin=0,
        xmax=EnvVariables.found_max_r_works,
        facecolor="k",
        alpha=0.1,
        zorder=-1,
    )
    ax.axhspan(
        ymin=0,
        ymax=EnvVariables.found_max_r_works,
        facecolor="k",
        alpha=0.1,
        zorder=-1,
    )

    fig.tight_layout()
    fig.savefig(
        figures_dir / f"{filename}.png",
        dpi=360,
        bbox_inches="tight",
    )
    plt.close()


def to_csv(
    pair_db_path: pathlib.Path,
    figures_dir: pathlib.Path,
    ligand_dir: pathlib.Path,
    filename: str,
) -> None:
    """Make plot."""
    dataframe = atomlite.Database(pair_db_path).get_property_df(
        properties=[
            "$.ligand1_key",
            "$.ligand2_key",
            "$.mean_s1",
            "$.min_s1",
            "$.mean_s2",
            "$.min_s2",
        ]
    )

    tdata = dataframe.with_columns(
        pl.Series(
            name="l1_smiles",
            values=[
                stk.Smiles().get_key(
                    stk.BuildingBlock.init_from_file(
                        ligand_dir / f"{l1key}_lowe.mol"
                    )
                )
                for l1key in dataframe["$.ligand1_key"]
            ],
        ),
        pl.Series(
            name="l2_smiles",
            values=[
                stk.Smiles().get_key(
                    stk.BuildingBlock.init_from_file(
                        ligand_dir / f"{l2key}_lowe.mol"
                    )
                )
                for l2key in dataframe["$.ligand2_key"]
            ],
        ),
    )

    tdata.write_csv(
        file=figures_dir / f"{filename}.csv",
        include_header=True,
    )


def get_pd_polymer() -> stk.BuildingBlock:
    """Define a building block."""
    palladium_atom = stk.BuildingBlock(
        smiles="[Pd+2]",
        functional_groups=(
            stk.SingleAtom(stk.Pd(0, charge=2)) for i in range(4)
        ),
        position_matrix=[[0.0, 0.0, 0.0]],
    )
    small_complex = stk.BuildingBlock(
        smiles="O",
        functional_groups=(
            stk.SmartsFunctionalGroupFactory(
                smarts="[#1]~[#8]~[#1]",
                bonders=(1,),
                deleters=(),
            ),
        ),
    )
    bb1 = stk.BuildingBlock(
        smiles=("CN=C"),
        functional_groups=[
            stk.SmartsFunctionalGroupFactory(
                smarts="[#6]~[#7X2]~[#6]",
                bonders=(1,),
                deleters=(),
            ),
        ],
    )
    polymer = stk.ConstructedMolecule(
        topology_graph=stk.metal_complex.SquarePlanar(
            metals=palladium_atom,
            ligands={bb1: (0, 2), small_complex: (1, 3)},
            optimizer=stk.MCHammer(target_bond_length=0.5),
        ),
    )

    return stk.BuildingBlock.init_from_molecule(
        molecule=polymer,
        functional_groups=stk.SmartsFunctionalGroupFactory(
            smarts="[#7]~[#46](~[#8](~[#1])~[#1])~[#7]",
            bonders=(1,),
            deleters=(2, 3, 4),
        ),
    )


class FakeMacro(stk.cage.Cage):
    """Fake a macrocycle for now."""

    _vertex_prototypes = (
        stk.cage.LinearVertex(
            0, np.array([0, 0.5, 0]), use_neighbor_placement=False
        ),
        stk.cage.LinearVertex(
            1, np.array([0, -0.5, 0]), use_neighbor_placement=False
        ),
        stk.cage.LinearVertex(
            2, np.array([1, 0, 0]), use_neighbor_placement=False
        ),
        stk.cage.LinearVertex(
            3, np.array([-1, 0, 0]), use_neighbor_placement=False
        ),
    )

    _edge_prototypes = (
        stk.Edge(0, _vertex_prototypes[0], _vertex_prototypes[2]),
        stk.Edge(1, _vertex_prototypes[0], _vertex_prototypes[3]),
        stk.Edge(2, _vertex_prototypes[1], _vertex_prototypes[2]),
        stk.Edge(3, _vertex_prototypes[1], _vertex_prototypes[3]),
    )


def merge_two_ligands(
    mol1: stk.BuildingBlock,
    mol2: stk.BuildingBlock,
    pair_key: str,
    state: int,
    struct_dir: pathlib.Path,
) -> stk.BuildingBlock:
    """Merge two ligands to be in one structure."""
    mol_file = struct_dir / f"{pair_key}_merged.mol"
    if not mol_file.exists():
        mol1 = bbprep.FurthestFGs().modify(
            building_block=stk.BuildingBlock.init_from_molecule(
                mol1,
                functional_groups=(
                    stk.SmartsFunctionalGroupFactory(
                        smarts="[#6]~[#7X2]~[#6]",
                        bonders=(1,),
                        deleters=(),
                    ),
                ),
            ),
            desired_functional_groups=2,
        )
        mol2 = bbprep.FurthestFGs().modify(
            building_block=stk.BuildingBlock.init_from_molecule(
                mol2,
                functional_groups=(
                    stk.SmartsFunctionalGroupFactory(
                        smarts="[#6]~[#7X2]~[#6]",
                        bonders=(1,),
                        deleters=(),
                    ),
                ),
            ),
            desired_functional_groups=2,
        )
        pd = get_pd_polymer()

        if state == 1:
            va = {0: 0, 1: 1}
        elif state == 2:  # noqa: PLR2004
            va = {0: 0, 1: 0}

        constructued = stk.ConstructedMolecule(
            topology_graph=FakeMacro(
                building_blocks={
                    mol1: (0,),
                    mol2: (1,),
                    pd: (2, 3),
                },
                optimizer=stk.MCHammer(num_steps=200),
                vertex_alignments=va,
                reaction_factory=stk.DativeReactionFactory(
                    stk.GenericReactionFactory(
                        bond_orders={
                            frozenset(
                                {
                                    stk.GenericFunctionalGroup,
                                    stk.SingleAtom,
                                }
                            ): 9,
                        },
                    ),
                ),
            ),
        )

        # Remove fluff.
        ligands = [
            i
            for i in stko.molecule_analysis.DecomposeMOC().decompose(
                molecule=constructued,
                metal_atom_nos=(46,),
            )
            if i.get_num_atoms() > 8  # noqa: PLR2004
        ]
        constructued = merge_stk_molecules(ligands)
        constructued.write(mol_file)

    else:
        constructued = stk.BuildingBlock.init_from_file(mol_file)

    return stk.BuildingBlock.init_from_molecule(constructued)


def merge_two_ligands_with_cg(  # noqa: PLR0913
    mol1: stk.BuildingBlock,
    mol2: stk.BuildingBlock,
    cgmol: stk.BuildingBlock,
    pair_key: str,
    state: int,
    struct_dir: pathlib.Path,
) -> stk.BuildingBlock:
    """Merge two ligands to be in one structure."""
    struct_dir.mkdir(exist_ok=True, parents=True)
    mol_file = struct_dir / f"{pair_key}_merged.mol"

    if not mol_file.exists():
        mol1 = bbprep.FurthestFGs().modify(
            building_block=stk.BuildingBlock.init_from_molecule(
                mol1,
                functional_groups=(
                    stko.functional_groups.ThreeSiteFactory(
                        smarts="[#6]~[#7X2]~[#6]",
                        bonders=(1,),
                        deleters=(),
                    ),
                ),
            ),
            desired_functional_groups=2,
        )
        mol2 = bbprep.FurthestFGs().modify(
            building_block=stk.BuildingBlock.init_from_molecule(
                mol2,
                functional_groups=(
                    stko.functional_groups.ThreeSiteFactory(
                        smarts="[#6]~[#7X2]~[#6]",
                        bonders=(1,),
                        deleters=(),
                    ),
                ),
            ),
            desired_functional_groups=2,
        )

        # First, we know that the model moves CG models into x-y plane at
        # specific origins.
        m1_origin = np.array((-10, 0, 0))
        m2_origin = np.array((4, 0, 0))
        nn_axis = np.array((0, 1, 0))

        mol1 = mol1.with_centroid(position=m1_origin)
        mol2 = mol2.with_centroid(position=m2_origin)

        # We know m1 faces [1,0,0] and m2 faces [-1, 0, 0]
        placer_centroid = mol1.get_centroid(atom_ids=mol1.get_placer_ids())
        core_centroid = mol1.get_centroid(atom_ids=mol1.get_core_atom_ids())
        core_to_placer = placer_centroid - core_centroid
        mol1 = mol1.with_rotation_between_vectors(
            start=core_to_placer,
            target=np.array([1, 0, 0]),
            origin=mol1.get_centroid(),
        )

        placer_centroid = mol2.get_centroid(atom_ids=mol2.get_placer_ids())
        core_centroid = mol2.get_centroid(atom_ids=mol2.get_core_atom_ids())
        core_to_placer = placer_centroid - core_centroid
        mol2 = mol2.with_rotation_between_vectors(
            start=core_to_placer,
            target=np.array([-1, 0, 0]),
            origin=mol2.get_centroid(),
        )

        # Then orient with original axis.
        fg0_position, fg1_position = (
            mol1.get_centroid(fg.get_placer_ids())
            for fg in mol1.get_functional_groups()
        )
        mol1 = mol1.with_rotation_between_vectors(
            start=fg1_position - fg0_position,
            target=nn_axis,
            origin=mol1.get_centroid(),
        )

        fg0_position, fg1_position = (
            mol2.get_centroid(fg.get_placer_ids())
            for fg in mol2.get_functional_groups()
        )
        mol2 = mol2.with_rotation_between_vectors(
            start=fg1_position - fg0_position,
            target=nn_axis if state == 1 else -nn_axis,
            origin=mol2.get_centroid(),
        )

        # Orient again into the xy plane.
        placer_centroid = mol1.get_centroid(atom_ids=mol1.get_placer_ids())
        core_centroid = mol1.get_centroid(atom_ids=mol1.get_core_atom_ids())
        core_to_placer = placer_centroid - core_centroid
        mol1 = mol1.with_rotation_between_vectors(
            start=core_to_placer,
            target=np.array([1, 0, 0]),
            origin=mol1.get_centroid(),
        )

        placer_centroid = mol2.get_centroid(atom_ids=mol2.get_placer_ids())
        core_centroid = mol2.get_centroid(atom_ids=mol2.get_core_atom_ids())
        core_to_placer = placer_centroid - core_centroid
        mol2 = mol2.with_rotation_between_vectors(
            start=core_to_placer,
            target=np.array([-1, 0, 0]),
            origin=mol2.get_centroid(),
        )

        # Orient again with new NN axis.
        hh1_ids = [
            i.get_id() for i in cgmol.get_atoms() if i.get_atomic_number() == 1
        ][:2]
        hh2_ids = [
            i.get_id() for i in cgmol.get_atoms() if i.get_atomic_number() == 1
        ][2:]
        fg0_position, fg1_position = (
            mol1.get_centroid(fg.get_placer_ids())
            for fg in mol1.get_functional_groups()
        )
        edge_position1, edge_position2 = list(
            cgmol.get_atomic_positions(hh1_ids)
        )
        mol1 = mol1.with_rotation_between_vectors(
            start=fg1_position - fg0_position,
            target=edge_position2 - edge_position1,
            origin=mol1.get_centroid(),
        )

        fg0_position, fg1_position = (
            mol2.get_centroid(fg.get_placer_ids())
            for fg in mol2.get_functional_groups()
        )
        edge_position1, edge_position2 = list(
            cgmol.get_atomic_positions(hh2_ids)
        )
        mol2 = mol2.with_rotation_between_vectors(
            start=fg1_position - fg0_position,
            target=edge_position2 - edge_position1
            if state == 1
            else edge_position1 - edge_position2,
            origin=mol2.get_centroid(),
        )

        constructued = merge_stk_molecules((mol1, mol2, cgmol))

        constructued.write(mol_file)

    else:
        constructued = stk.BuildingBlock.init_from_file(mol_file)

    return stk.BuildingBlock.init_from_molecule(constructued)


def to_chemiscope(
    pair_db_path: pathlib.Path,
    figures_dir: pathlib.Path,
    struct_dir: pathlib.Path,
    ligand_dir: pathlib.Path,
    filename: str,
) -> None:
    """Make plot."""
    structures = []
    properties = {
        "key": [],
        "mean_s1": [],
        "mean_s2": [],
        "min_s1": [],
        "min_s2": [],
    }

    pair_db = atomlite.Database(pair_db_path)
    for entry in pair_db.get_entries():
        try:
            molecule = stk.BuildingBlock.init_from_rdkit_mol(
                atomlite.json_to_rdkit(entry.molecule)
            )

        except RuntimeError:
            # Incase I had used the old algorithm that gives 2D structures.
            mjson = entry.molecule

            new_confs = [[*i, 0] for i in mjson["conformers"][0]]

            mjson["conformers"][0] = new_confs
            pair_db.update_entries(
                atomlite.Entry(key=entry.key, molecule=mjson)
            )
            molecule = stk.BuildingBlock.init_from_rdkit_mol(
                atomlite.json_to_rdkit(pair_db.get_entry(entry.key).molecule)
            )

        l1 = entry.properties["ligand1_key"]
        l2 = entry.properties["ligand2_key"]
        s1cid1 = entry.properties["best_s1_cid1"]
        s1cid2 = entry.properties["best_s1_cid2"]
        s2cid1 = entry.properties["best_s2_cid1"]
        s2cid2 = entry.properties["best_s2_cid2"]

        cgmolecule_tuple = (
            (
                stk.BuildingBlock.init_from_file(
                    ligand_dir / f"confs_{l1}" / f"{l1}_c{s1cid1}_cuff.mol"
                ),
                stk.BuildingBlock.init_from_file(
                    ligand_dir / f"confs_{l2}" / f"{l2}_c{s1cid2}_cuff.mol"
                ),
                1,
            )
            if entry.properties["best_s1_residual"]
            < entry.properties["best_s2_residual"]
            else (
                stk.BuildingBlock.init_from_file(
                    ligand_dir / f"confs_{l1}" / f"{l1}_c{s2cid1}_cuff.mol"
                ),
                stk.BuildingBlock.init_from_file(
                    ligand_dir / f"confs_{l2}" / f"{l2}_c{s2cid2}_cuff.mol"
                ),
                2,
            )
        )

        structures.append(
            merge_two_ligands_with_cg(
                mol1=cgmolecule_tuple[0],
                mol2=cgmolecule_tuple[1],
                cgmol=molecule,
                pair_key=entry.key,
                state=cgmolecule_tuple[2],
                struct_dir=struct_dir,
            )
        )

        properties["mean_s1"].append(entry.properties["mean_s1"])
        properties["mean_s2"].append(entry.properties["mean_s2"])
        properties["min_s1"].append(entry.properties["min_s1"])
        properties["min_s2"].append(entry.properties["min_s2"])
        properties["key"].append(entry.key)

    shape_dict = chemiscope.convert_stk_bonds_as_shapes(
        frames=structures,
        bond_color="#000000",
        bond_radius=0.12,
    )

    # Write the shape string for settings to turn them on automatically.
    shape_string = ",".join(shape_dict.keys())

    chemiscope.write_input(
        path=str(figures_dir / f"{filename}.json.gz"),
        frames=structures,
        properties=properties,
        meta={"name": "Candidates."},
        settings=chemiscope.quick_settings(
            x="mean_s1",
            y="mean_s2",
            color="",
            structure_settings={
                "shape": shape_string,
                "atoms": True,
                "bonds": False,
                "spaceFilling": False,
            },
        ),
        shapes=shape_dict,
    )


def to_cg_chemiscope(
    pair_db_path: pathlib.Path,
    figures_dir: pathlib.Path,
    filename: str,
) -> None:
    """Make plot."""
    structures = []
    properties = {
        "key": [],
        "mean_s1": [],
        "mean_s2": [],
        "min_s1": [],
        "min_s2": [],
    }
    pair_db = atomlite.Database(pair_db_path)
    for entry in pair_db.get_entries():
        try:
            molecule = stk.BuildingBlock.init_from_rdkit_mol(
                atomlite.json_to_rdkit(entry.molecule)
            )

        except RuntimeError:
            # Incase I had used the old algorithm that gives 2D structures.
            mjson = entry.molecule

            new_confs = [[*i, 0] for i in mjson["conformers"][0]]

            mjson["conformers"][0] = new_confs
            pair_db.update_entries(
                atomlite.Entry(key=entry.key, molecule=mjson)
            )
            molecule = stk.BuildingBlock.init_from_rdkit_mol(
                atomlite.json_to_rdkit(pair_db.get_entry(entry.key).molecule)
            )

        structures.append(molecule)

        properties["mean_s1"].append(entry.properties["mean_s1"])
        properties["mean_s2"].append(entry.properties["mean_s2"])
        properties["min_s1"].append(entry.properties["min_s1"])
        properties["min_s2"].append(entry.properties["min_s2"])
        properties["key"].append(entry.key)

    shape_dict = chemiscope.convert_stk_bonds_as_shapes(
        frames=structures,
        bond_color="#000000",
        bond_radius=0.12,
    )

    # Write the shape string for settings to turn them on automatically.
    shape_string = ",".join(shape_dict.keys())

    chemiscope.write_input(
        path=str(figures_dir / f"{filename}.json.gz"),
        frames=structures,
        properties=properties,
        meta={"name": "CG- Candidates."},
        settings=chemiscope.quick_settings(
            x="mean_s1",
            y="mean_s2",
            color="",
            structure_settings={
                "shape": shape_string,
                "atoms": True,
                "bonds": False,
                "spaceFilling": False,
            },
        ),
        shapes=shape_dict,
    )
