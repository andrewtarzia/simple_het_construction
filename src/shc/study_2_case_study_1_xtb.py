"""Script to build the ligands in this case study."""

import json
import logging
import pathlib
import subprocess as sp
from pathlib import Path

import atomlite
import bbprep
import matplotlib.pyplot as plt
import numpy as np
import stk
import stko
from rdkit.Chem import AllChem as rdkit  # noqa: N813

from shc.definitions import EnvVariables, Study1EnvVariables


class ConstrainedXTBFF(stko.XTBFF):
    """Adds constraints to XTBFF optimisation."""

    def _write_detailed_control(self, constraints: tuple[str, ...]) -> None:
        if len(constraints) != 11:  # noqa: PLR2004
            raise SystemExit
        constrain_str = "$constrain\n"
        for const in constraints:
            if len(const) == 3:  # noqa: PLR2004
                front = "    angle: "
                end = ", 90\n"
                constrain_str += f"{front}{const[0] + 1}, {const[1] + 1}, {const[2] + 1}{end}"
            elif len(const) == 4:  # noqa: PLR2004
                front = "    dihedral: "
                end = ", 0\n"
                constrain_str += (
                    f"{front}{const[0] + 1}, {const[1] + 1}, {const[2] + 1},"
                    f" {const[3] + 1}{end}"
                )

            else:
                raise RuntimeError

        constrain_str += "$end\n"

        with Path("det_control.in").open("w") as f:
            f.write(constrain_str)

    def _run_xtb(self, xyz: str, out_file: Path | str) -> None:
        """Run GFN-xTB.

        Parameters:
            xyz:
                The name of the input structure ``.xyz`` file.

            out_file:
                The name of output file with xTB results.


        """
        out_file = Path(out_file)

        # Modify the memory limit.
        memory = "ulimit -s unlimited ;" if self._unlimited_memory else ""

        # Set optimization level and type.
        optimization = f"--opt {self._opt_level}"

        cmd = (
            f"{memory} {self._xtb_path} {xyz} "
            f"--gfnff "
            f"{optimization} --parallel {self._num_cores} "
            f"--chrg {self._charge} -I det_control.in"
        )

        with out_file.open("w") as f:
            # Note that sp.call will hold the program until completion
            # of the calculation.
            sp.call(  # noqa: S602
                cmd,
                stdin=sp.PIPE,
                stdout=f,
                stderr=sp.PIPE,
                # Shell is required to run complex arguments.
                shell=True,
            )

    def _get_constraints(self, mol: stk.Molecule) -> tuple[tuple, ...]:
        constraints = []

        with_pyridines = stk.BuildingBlock.init_from_molecule(
            molecule=mol,
            functional_groups=(
                stk.SmartsFunctionalGroupFactory(
                    smarts="[#6H2]~[#7X3](~[#46])~[#6H3]",
                    bonders=(1,),
                    deleters=(),
                ),
            ),
        )

        with_all_ligands = stk.BuildingBlock.init_from_molecule(
            molecule=mol,
            functional_groups=(
                stk.SmartsFunctionalGroupFactory(
                    smarts="[#6]~[#7X3](~[#46])~[#6]",
                    bonders=(1,),
                    deleters=(),
                ),
            ),
        )
        if with_pyridines.get_num_functional_groups() != 4:  # noqa: PLR2004
            raise RuntimeError
        if with_all_ligands.get_num_functional_groups() != 8:  # noqa: PLR2004
            raise RuntimeError

        ns_in_pyr = tuple(
            i.get_id()
            for fg in with_pyridines.get_functional_groups()
            for i in fg.get_atoms()
            if i.get_atomic_number() == 7  # noqa: PLR2004
        )
        ns_in_all = tuple(
            i.get_id()
            for fg in with_all_ligands.get_functional_groups()
            for i in fg.get_atoms()
            if i.get_atomic_number() == 7  # noqa: PLR2004
        )
        ns_in_lig = tuple(i for i in ns_in_all if i not in ns_in_pyr)

        # Get N-Pd-N angles.
        for a_ids in rdkit.FindAllPathsOfLengthN(
            mol=mol.to_rdkit_mol(),
            length=3,
            useBonds=False,
            useHs=False,
        ):
            atom_ids = list(a_ids)
            atoms = list(mol.get_atoms(atom_ids=a_ids))
            atom1 = atoms[0]
            atom2 = atoms[1]
            atom3 = atoms[2]
            angle_type = (
                atom1.__class__.__name__,
                atom2.__class__.__name__,
                atom3.__class__.__name__,
            )
            if (
                (angle_type == ("N", "Pd", "N")
                and (atom_ids[0] in ns_in_pyr and atom_ids[2] in ns_in_lig))
                or (atom_ids[2] in ns_in_pyr and atom_ids[0] in ns_in_lig)
            ):
                constraints.append(tuple(atom_ids))

        # Do some fancy (actually hard-coded...) atom selection to get the
        # torsions.
        constraint_options = []

        pd1 = constraints[0][1]
        pd2 = next(i[1] for i in constraints if i[1] != pd1)
        n11 = (
            constraints[0][0]
            if constraints[0][0] in ns_in_lig
            else constraints[0][2]
        )
        n12_options = [
            i for i in constraints if i[1] == pd1 and n11 not in (i[0], i[2])
        ]
        n12 = (
            n12_options[0][0]
            if n12_options[0][0] in ns_in_lig
            else n12_options[0][2]
        )
        n2_options = [i for i in constraints if i[1] != pd1]
        n21 = (
            n2_options[0][0]
            if n2_options[0][0] in ns_in_lig
            else n2_options[0][2]
        )
        n22_options = [i for i in n2_options if n21 not in (i[0], i[2])]
        n22 = (
            n22_options[0][0]
            if n22_options[0][0] in ns_in_lig
            else n22_options[0][2]
        )

        pos_mat = mol.get_position_matrix()
        # Get Pd-N-N-Pd angles.
        constraint_options.append(((pd1, n11, n21, pd2), (pd1, n11, n22, pd2)))
        constraint_options.append(((pd1, n12, n22, pd2), (pd1, n12, n21, pd2)))
        # Get N-N-N-N torsion.
        constraint_options.append(((n11, n12, n22, n21), (n11, n12, n21, n22)))
        for const1, const2 in constraint_options:
            torsion1 = abs(
                stko.calculate_dihedral(
                    pt1=pos_mat[const1[0]],
                    pt2=pos_mat[const1[1]],
                    pt3=pos_mat[const1[2]],
                    pt4=pos_mat[const1[3]],
                )
            )
            torsion2 = abs(
                stko.calculate_dihedral(
                    pt1=pos_mat[const2[0]],
                    pt2=pos_mat[const2[1]],
                    pt3=pos_mat[const2[2]],
                    pt4=pos_mat[const2[3]],
                )
            )

            nearest_zero = min((torsion1, torsion2))

            if torsion1 == nearest_zero:
                chosen_torsion = const1
            elif torsion2 == nearest_zero:
                chosen_torsion = const2

            if chosen_torsion in constraints:
                msg = "torsion already saved!"
                raise RuntimeError(msg)
            constraints.append(chosen_torsion)

        return tuple(constraints)

    def _run_optimization(
        self,
        mol: stk.Molecule,
    ) -> tuple[stk.Molecule, bool]:
        """Run loop of optimizations on `mol` using xTB.

        Parameters:
            mol: The molecule to be optimized.

        Returns:
            The optimized molecule and ``True`` if the calculation
            is complete or ``False`` if the calculation is incomplete.

        """
        xyz = "input_structure_ff.xyz"
        out_file = "optimization_ff.output"
        mol.write(xyz)

        constraints = self._get_constraints(mol)
        self._write_detailed_control(constraints)
        self._run_xtb(xyz=xyz, out_file=out_file)

        # Check if the optimization is complete.
        output_xyz = "xtbopt.xyz"
        opt_complete = self._is_complete(out_file)
        mol = mol.with_structure_from_file(output_xyz)

        return mol, opt_complete


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


def get_pd_benzene_polymer() -> stk.BuildingBlock:
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


def analyse_ligand_pair(  # noqa: PLR0913
    ligand1: str,
    ligand2: str,
    key: str,
    ligand_db: atomlite.Database,
    pair_db: atomlite.Database,
    calculation_dir: pathlib.Path,
) -> None:
    """Analyse a pair of ligands."""
    ligand1_entry = ligand_db.get_entry(ligand1)
    ligand2_entry = ligand_db.get_entry(ligand2)

    ligand1_min_energy = ligand1_entry.properties["energy;au"]
    ligand2_min_energy = ligand2_entry.properties["energy;au"]

    ligand1_molecule = bbprep.FurthestFGs().modify(
        building_block=stk.BuildingBlock.init_from_rdkit_mol(
            atomlite.json_to_rdkit(ligand1_entry.molecule),
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
    ligand2_molecule = bbprep.FurthestFGs().modify(
        building_block=stk.BuildingBlock.init_from_rdkit_mol(
            atomlite.json_to_rdkit(ligand2_entry.molecule),
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

    ligand1_smiles = stk.Smiles().get_key(ligand1_molecule)
    ligand2_smiles = stk.Smiles().get_key(ligand2_molecule)

    pd_benzene_polymer = get_pd_benzene_polymer()

    ligand_strains = {"state1": {}, "state2": {}}
    for state, va in zip(
        ("state1", "state2"), ({0: 0, 1: 0}, {0: 0, 1: 1}), strict=False
    ):
        macrocycle = stk.ConstructedMolecule(
            topology_graph=FakeMacro(
                building_blocks={
                    ligand1_molecule: (0,),
                    ligand2_molecule: (1,),
                    pd_benzene_polymer: (2, 3),
                },
                optimizer=stk.MCHammer(),
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
        macrocycle.write(calculation_dir / f"{key}_{state}_unopt.mol")

        preopt_output = calculation_dir / f"{key}_{state}_gulp1.mol"
        if not preopt_output.exists():
            output_dir = calculation_dir / f"{key}_{state}_gulp1"

            logging.info("UFF4MOF optimisation of %s in %s", key, state)
            gulp_opt = stko.GulpUFFOptimizer(
                gulp_path=EnvVariables.gulp_path,
                maxcyc=1000,
                metal_FF={46: "Pd4+2"},
                metal_ligand_bond_order="",
                output_dir=output_dir,
                conjugate_gradient=True,
            )
            gulp_opt.assign_FF(macrocycle)
            gulp1_mol = gulp_opt.optimize(mol=macrocycle)

            gulp_opt = stko.GulpUFFOptimizer(
                gulp_path=EnvVariables.gulp_path,
                maxcyc=1000,
                metal_FF={46: "Pd4+2"},
                metal_ligand_bond_order="",
                output_dir=output_dir,
                conjugate_gradient=False,
            )
            gulp_opt.assign_FF(gulp1_mol)
            gulp2_mol = gulp_opt.optimize(mol=gulp1_mol)
            gulp2_mol.write(preopt_output)

        else:
            gulp2_mol = macrocycle.with_structure_from_file(preopt_output)

        try:
            gfnffopt_output = calculation_dir / f"{key}_{state}_xtbffcons.mol"
            if not gfnffopt_output.exists():
                output_dir = calculation_dir / f"{key}_{state}_xtbffcons"

                logging.info("XTBFF optimisation of %s in %s", key, state)
                optimiser = ConstrainedXTBFF(
                    xtb_path=EnvVariables.xtb_path,
                    output_dir=output_dir,
                    opt_level="crude",
                    num_cores=8,
                    charge=4,
                    unlimited_memory=True,
                )
                gfnffmol = optimiser.optimize(gulp2_mol)
                gfnffmol.write(gfnffopt_output)

            else:
                gfnffmol = macrocycle.with_structure_from_file(gfnffopt_output)
            failed = False
        except (stko.OptimizerError, stko.ConvergenceError):
            failed = True

        if failed:
            ligand_strains[state]["lig1"] = -1
            ligand_strains[state]["lig2"] = -1

        else:
            organic_linkers = stko.molecule_analysis.DecomposeMOC().decompose(
                molecule=gfnffmol,
                metal_atom_nos=(46,),
            )

            for ol in organic_linkers:
                smi = stk.Smiles().get_key(ol)
                if smi == ligand1_smiles:
                    energy = calculate_xtb_energy(
                        molecule=ol,
                        name=f"{key}_{state}_lig1",
                        charge=0,
                        calc_dir=calculation_dir,
                    )
                    ligand_strains[state]["lig1"] = energy - ligand1_min_energy

                elif smi == ligand2_smiles:
                    energy = calculate_xtb_energy(
                        molecule=ol,
                        name=f"{key}_{state}_lig2",
                        charge=0,
                        calc_dir=calculation_dir,
                    )
                    ligand_strains[state]["lig2"] = energy - ligand2_min_energy

    pair_db.set_property(
        key=key,
        path="$.pair_data.state_1_strain_1",
        property=float(ligand_strains["state1"]["lig1"]),
        commit=False,
    )
    pair_db.set_property(
        key=key,
        path="$.pair_data.state_1_strain_2",
        property=float(ligand_strains["state1"]["lig2"]),
        commit=False,
    )
    pair_db.set_property(
        key=key,
        path="$.pair_data.state_2_strain_1",
        property=float(ligand_strains["state2"]["lig1"]),
        commit=False,
    )
    pair_db.set_property(
        key=key,
        path="$.pair_data.state_2_strain_2",
        property=float(ligand_strains["state2"]["lig2"]),
        commit=False,
    )

    pair_db.connection.commit()


def unsymmetric_plot(
    pts: str,
    plot_targets: tuple[tuple[str, str], ...],
    pair_db: atomlite.Database,
    figures_dir: pathlib.Path,
) -> None:
    """Make plot."""
    fig, axs = plt.subplots(ncols=3, figsize=(16, 5))
    ax, ax1, ax2 = axs
    steps = range(len(plot_targets) - 1, -1, -1)
    for i, (ligand1, ligand2) in zip(steps, plot_targets, strict=False):
        key = f"{ligand1}_{ligand2}"
        entry = pair_db.get_property_entry(key)

        print(entry.properties)
        raise SystemExit

        xdata = [
            entry.properties["pair_data"][i]["state_1_residual"]
            for i in entry.properties["pair_data"]
        ]
        xmin = 0
        xmax = 15
        xwidth = 0.5
        xbins = np.arange(xmin - xwidth, xmax + xwidth, xwidth)
        ystep = 1
        ax.hist(
            x=xdata,
            bins=xbins,
            density=True,
            bottom=i * ystep,
            histtype="stepfilled",
            stacked=True,
            linewidth=1.0,
            alpha=1.0,
            edgecolor="k",
            label=f"{entry.key}",
        )
        ax.plot(
            (np.mean(xdata), np.mean(xdata)),
            ((i + 1) * ystep, i * ystep),
            alpha=1.0,
            c="k",
        )

        xdata = [
            entry.properties["pair_data"][i]["state_2_residual"]
            for i in entry.properties["pair_data"]
        ]
        xmin = 0
        xmax = 15
        xwidth = 0.5
        xbins = np.arange(xmin - xwidth, xmax + xwidth, xwidth)
        ystep = 1
        ax1.hist(
            x=xdata,
            bins=xbins,
            density=True,
            bottom=i * ystep,
            histtype="stepfilled",
            stacked=True,
            linewidth=1.0,
            alpha=1.0,
            edgecolor="k",
            label=f"{entry.key}",
        )
        ax1.plot(
            (np.mean(xdata), np.mean(xdata)),
            ((i + 1) * ystep, i * ystep),
            alpha=1.0,
            c="k",
        )

        xdata = [
            entry.properties["pair_data"][i]["state_1_residual"]
            - entry.properties["pair_data"][i]["state_2_residual"]
            for i in entry.properties["pair_data"]
        ]
        xmin = -1
        xmax = 1
        xwidth = 0.05
        xbins = np.arange(xmin - xwidth, xmax + xwidth, xwidth)
        ystep = 10
        ax2.hist(
            x=xdata,
            bins=xbins,
            density=True,
            bottom=i * ystep,
            histtype="stepfilled",
            stacked=True,
            linewidth=1.0,
            alpha=1.0,
            edgecolor="k",
            label=f"{entry.key}",
        )
        ax2.plot(
            (np.mean(xdata), np.mean(xdata)),
            ((i + 1) * ystep, i * ystep),
            alpha=1.0,
            c="k",
        )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel("1-residuals", fontsize=16)
    ax.set_ylabel("frequency", fontsize=16)
    ax.set_yticks([])
    ax.set_ylim(0, (steps[0] + 1.5) * 1)

    ax1.tick_params(axis="both", which="major", labelsize=16)
    ax1.set_xlabel("2-residuals", fontsize=16)
    ax1.set_yticks([])
    ax1.set_ylim(0, (steps[0] + 1.5) * 1)

    ax2.tick_params(axis="both", which="major", labelsize=16)
    ax2.set_xlabel("delta-residuals", fontsize=16)
    ax2.set_yticks([])
    ax2.set_ylim(0, (steps[0] + 1.5) * 10)
    ax2.legend(fontsize=16)

    fig.tight_layout()
    fig.savefig(
        figures_dir / f"cs1_residuals_{pts}.png",
        dpi=360,
        bbox_inches="tight",
    )
    plt.close()


def symmetric_plot(
    pts: str,
    plot_targets: tuple[tuple[str, str], ...],
    pair_db: atomlite.Database,
    figures_dir: pathlib.Path,
) -> None:
    """Make plot."""
    experimental_ligand_outcomes = {
        ("e16_0", "e10_0"): True,
        ("e16_0", "e17_0"): True,
        ("e10_0", "e17_0"): False,
        ("e11_0", "e10_0"): True,
        ("e16_0", "e14_0"): True,
        ("e18_0", "e14_0"): True,
        ("e18_0", "e10_0"): True,
        ("e12_0", "e10_0"): True,
        ("e11_0", "e14_0"): True,
        ("e12_0", "e14_0"): True,
        ("e11_0", "e13_0"): True,
        ("e12_0", "e13_0"): True,
        ("e13_0", "e14_0"): False,
        ("e11_0", "e12_0"): False,
        ("sla_0", "sl1_0"): False,
        ("slb_0", "sl1_0"): True,
        ("slc_0", "sl1_0"): True,
        ("sld_0", "sl1_0"): False,
        ("sla_0", "sl2_0"): False,
        ("slb_0", "sl2_0"): False,
        ("slc_0", "sl2_0"): False,
        ("sld_0", "sl2_0"): False,
        ("sla_0", "sl3_0"): False,
        ("slb_0", "sl3_0"): False,
        ("slc_0", "sl3_0"): False,
        ("sld_0", "sl3_0"): False,
    }

    def get_gvalues(result_dict: dict) -> list[float]:
        """Get Gavg from a dictionary of study 1 format."""
        geom_scores = []
        for cid_pair in result_dict:
            if (
                abs(result_dict[cid_pair]["large_dihedral"])
                > Study1EnvVariables.dihedral_cutoff
                or abs(result_dict[cid_pair]["small_dihedral"])
                > Study1EnvVariables.dihedral_cutoff
            ):
                continue

            geom_score = result_dict[cid_pair]["geom_score"]
            geom_scores.append(geom_score)
        return geom_scores

    study1_pair_file = (
        pathlib.Path("/home/atarzia/workingspace/cpl/study_1")
        / "all_pair_res.json"
    )
    with study1_pair_file.open("r") as f:
        pair_info = json.load(f)

    fig, ax1 = plt.subplots(figsize=(8, 5))

    steps = range(len(plot_targets) - 1, -1, -1)

    for _, (ligand1, ligand2) in zip(steps, plot_targets, strict=False):
        key = f"{ligand1}_{ligand2}"
        if "e" in ligand1:
            study1_key = f"{ligand1.split('_')[0]},{ligand2.split('_')[0]}"
        else:
            study1_key = (
                f"{ligand1.split('_')[0].split('s')[1]},"
                f"{ligand2.split('_')[0].split('s')[1]}"
            )
        if study1_key not in pair_info:
            if "e" in ligand1:
                study1_key = f"{ligand2.split('_')[0]},{ligand1.split('_')[0]}"
            else:
                study1_key = (
                    f"{ligand2.split('_')[0].split('s')[1]},"
                    f"{ligand1.split('_')[0].split('s')[1]}"
                )
            if study1_key not in pair_info:
                msg = f"{study1_key} not in pair_info: {pair_info.keys()}"
                raise RuntimeError(msg)

        works = experimental_ligand_outcomes[(ligand1, ligand2)]

        entry = pair_db.get_property_entry(key)
        print(entry.properties["pair_data"])
        if (
            entry.properties["pair_data"]["state_1_strain_1"] != -1
            and entry.properties["pair_data"]["state_1_strain_2"] != -1
        ):
            sum_strain = (
                sum(
                    (
                        entry.properties["pair_data"]["state_1_strain_1"],
                        entry.properties["pair_data"]["state_1_strain_2"],
                    )
                )
                * 2625.5
            )
        elif (
            entry.properties["pair_data"]["state_2_strain_1"] != -1
            and entry.properties["pair_data"]["state_2_strain_2"] != -1
        ):
            sum_strain = (
                sum(
                    (
                        entry.properties["pair_data"]["state_2_strain_1"],
                        entry.properties["pair_data"]["state_2_strain_2"],
                    )
                )
                * 2625.5
            )
        else:
            continue

        print(sum_strain)
        gvalues = get_gvalues(pair_info[study1_key])

        ax1.scatter(
            sum_strain,
            np.mean(gvalues),
            alpha=1.0,
            ec="k",
            c="tab:blue" if works else "tab:orange",
            marker="o" if works else "X",
            s=120,
        )

    ax1.tick_params(axis="both", which="major", labelsize=16)
    ax1.set_xlabel("sum strain [kjmol-1]", fontsize=16)
    ax1.set_ylabel("g_avg study 1", fontsize=16)
    # ax1.set_xlim(0, None)
    ax1.set_ylim(0, 1)
    ax1.set_xscale("log")

    fig.tight_layout()
    fig.savefig(
        figures_dir / f"cs1_residuals_{pts}.png",
        dpi=360,
        bbox_inches="tight",
    )
    plt.close()


def calculate_xtb_energy(
    molecule: stk.Molecule,
    name: str,
    charge: int,
    calc_dir: pathlib.Path,
) -> float:
    """Calculate energy."""
    output_dir = calc_dir / f"{name}_xtbey"
    output_file = calc_dir / f"{name}_xtb.ey"

    if output_file.exists():
        with output_file.open("r") as f:
            lines = f.readlines()
        for line in lines:
            energy = float(line.rstrip())
            break
    else:
        logging.info("xtb energy calculation of %s", name)
        xtb = stko.XTBEnergy(
            xtb_path=EnvVariables.xtb_path,
            output_dir=output_dir,
            gfn_version=2,
            num_cores=6,
            charge=charge,
            num_unpaired_electrons=0,
            unlimited_memory=True,
        )
        energy = xtb.get_energy(mol=molecule)
        with output_file.open("w") as f:
            f.write(f"{energy}\n")

    # In a.u.
    return energy


def get_lowest_energy_xtb_conformer(
    molecule: stk.Molecule,
    name: str,
    charge: int,
    calc_dir: pathlib.Path,
    ligand_db: atomlite.Database,
) -> None:
    """Use ligand ensemble and xtb to get lowest energy conformer."""
    low_energy_structure = calc_dir / f"{name}_lowe_xtbff.mol"
    rxn_structure = calc_dir / f"{name}_rxn.mol"

    generator = bbprep.generators.ETKDG(num_confs=100)
    ensemble = generator.generate_conformers(molecule)
    min_energy = float("inf")
    min_conformer = None
    min_n_centroid_n_angle = float("inf")
    rxn_conformer = None
    for conformer in ensemble.yield_conformers():
        conf_name = f"{name}_{conformer.conformer_id}"
        xtbffopt_output = calc_dir / f"{conf_name}_xtbff.mol"
        try:
            if not xtbffopt_output.exists():
                output_dir = calc_dir / f"{conf_name}_xtbffopt"
                logging.info("    GFN-FF optimisation of %s", conf_name)
                xtbff_opt = stko.XTBFF(
                    xtb_path=EnvVariables.xtb_path,
                    output_dir=output_dir,
                    num_cores=6,
                    charge=charge,
                    opt_level="normal",
                )
                xtbffopt_mol = xtbff_opt.optimize(mol=conformer.molecule)
                xtbffopt_mol.write(xtbffopt_output)
            else:
                xtbffopt_mol = conformer.molecule.with_structure_from_file(
                    str(xtbffopt_output)
                )
        except stko.ConvergenceError:
            continue

        energy = calculate_xtb_energy(
            molecule=xtbffopt_mol,
            name=conf_name,
            charge=charge,
            calc_dir=calc_dir,
        )
        if energy < min_energy:
            min_energy = energy
            min_conformer = bbprep.Conformer(
                molecule=xtbffopt_mol,
                conformer_id=conformer.conformer_id,
                source=conformer.source,
                permutation=None,
            )

        analyser = stko.molecule_analysis.DitopicThreeSiteAnalyser()
        n_centroid_n_angle = analyser.get_binder_centroid_angle(
            molecule=bbprep.FurthestFGs().modify(
                building_block=stk.BuildingBlock.init_from_molecule(
                    molecule=xtbffopt_mol,
                    functional_groups=(stko.functional_groups.CNCFactory(),),
                ),
                desired_functional_groups=2,
            ),
        )

        if n_centroid_n_angle < min_n_centroid_n_angle:
            min_n_centroid_n_angle = n_centroid_n_angle
            rxn_conformer = bbprep.Conformer(
                molecule=xtbffopt_mol,
                conformer_id=conformer.conformer_id,
                source=conformer.source,
                permutation=None,
            )

    logging.info(
        "lowest energy for %s is %s, rxn is %s",
        name,
        min_conformer.conformer_id,
        rxn_conformer.conformer_id,
    )

    min_conformer.molecule.write(low_energy_structure)
    rxn_conformer.molecule.write(rxn_structure)
    entry = atomlite.Entry.from_rdkit(
        key=name,
        molecule=rxn_conformer.molecule.to_rdkit_mol(),
        properties={"energy;au": min_energy},
    )
    ligand_db.add_entries(entry)


def main() -> None:
    """Run script."""
    ligand_dir = pathlib.Path("/home/atarzia/workingspace/cpl/cs1xtb_ligands")
    calculation_dir = pathlib.Path(
        "/home/atarzia/workingspace/cpl/cs1xtb_calculations"
    )
    figures_dir = pathlib.Path(
        "/home/atarzia/workingspace/cpl/figures/cs1xtb/"
    )
    ligand_dir.mkdir(exist_ok=True)
    calculation_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True, parents=True)

    ligand_db = atomlite.Database(ligand_dir / "cs1xtb_ligands.db")
    pair_db = atomlite.Database(ligand_dir / "cs1xtb_pairs.db")

    # Build all ligands.
    ligand_smiles = {
        ##### Converging. #####
        # From Chand.
        "lab_0": "C1=CC=NC=C1C1=CC(C#CC2=CC(C(NC3=CN=CC=C3)=O)=CC=C2)=CC=C1",
        # From molinksa.
        "m2h_0": "C1=CC(=CC(=C1)C#CC2=CN=CC=C2)C#CC3=CN=CC=C3",
        # From study 1.
        "sl1_0": "C1=NC=CC(C2=CC=C3OC4C=CC(C5C=CN=CC=5)=CC=4C3=C2)=C1",
        "sl2_0": "C1=CC(=CC(=C1)C2=CC=NC=C2)C3=CC=NC=C3",
        "sl3_0": "C1=CN=CC=C1C2=CC=C(S2)C3=CC=NC=C3",
        ##### Diverging. #####
        # From Chand.
        "la_0": ("C1=NC=C2C=CC=C(C#CC3=CC=CN=C3)C2=C1"),
        "lb_0": (
            "C([H])1C([H])=C2C(C([H])=C([H])C([H])=C2C2=C([H])C([H])=C(C3C([H]"
            ")=NC([H])=C([H])C=3[H])C([H])=C2[H])=C([H])N=1"
        ),
        "lc_0": ("C1=CC(=CN=C1)C#CC2=CN=CC=C2"),
        "ld_0": ("C1=CC(=CN=C1)C2=CC=C(C=C2)C3=CN=CC=C3"),
        # From molinksa.
        "m4q_0": "C1=CC=C2C(=C1)C=C(C=N2)C#CC3=CN=CC=C3",
        "m4p_0": "C1=CN=CC(C#CC2=CN=C(C)C=C2)=C1",
        # From study 1.
        "sla_0": (
            "C1=CN=CC2C(C3=CC=C(C#CC4=CC5C6C=C(C#CC7=CC=C(C8=CC=CC9C=C"
            "N=CC8=9)C=C7)C=CC=6OC=5C=C4)C=C3)=CC=CC1=2"
        ),
        "slb_0": (
            "C1=CN=CC2C(C3=CC=C(C#CC4N=C(C#CC5=CC=C(C6=CC=CC7C=CN=CC6="
            "7)C=C5)C=CC=4)C=C3)=CC=CC1=2"
        ),
        "slc_0": (
            "C1C2=C(C(=CC=C2)C2C=CC(C#CC3=CC=CC(C#CC4C=CC(C5C6=C(C=CN="
            "C6)C=CC=5)=CC=4)=C3)=CC=2)C=NC=1"
        ),
        "sld_0": (
            "C1C2=C(C(=CC=C2)C2C=CC(C#CC3=CC=C(C#CC4C=CC(C5C6=C(C=CN=C"
            "6)C=CC=5)=CC=4)S3)=CC=2)C=NC=1"
        ),
        # Experimental.
        "e10_0": (
            "C1=CC(C#CC2=CC3C4C=C(C#CC5=CC=CN=C5)C=CC=4N(C)C=3C=C2)=CN=C1"
        ),
        "e11_0": "C1N=CC=CC=1C1=CC2=C(C3=C(C2(C)C)C=C(C2=CN=CC=C2)C=C3)C=C1",
        "e12_0": "C1=CC=C(C2=CC3C(=O)C4C=C(C5=CN=CC=C5)C=CC=4C=3C=C2)C=N1",
        "e13_0": (
            "C1C=C(N2C(=O)C3=C(C=C4C(=C3)C3(C5=C(C4(C)CC3)C=C3C(C(N(C3="
            "O)C3C=CC=NC=3)=O)=C5)C)C2=O)C=NC=1"
        ),
        "e14_0": (
            "C1=CN=CC(C#CC2C=CC3C(=O)C4C=CC(C#CC5=CC=CN=C5)=CC=4C=3C=2)=C1"
        ),
        "e16_0": (
            "C(C1=CC2C3C=C(C4=CC=NC=C4)C=CC=3C(OC)=C(OC)C=2C=C1)1=CC=NC=C1"
        ),
        "e17_0": (
            "C12C=CN=CC=1C(C#CC1=CC=C3C(C(C4=C(N3C)C=CC(C#CC3=CC=CC5C3="
            "CN=CC=5)=C4)=O)=C1)=CC=C2"
        ),
        "e18_0": (
            "C1(=CC=NC=C1)C#CC1=CC2C3C=C(C#CC4=CC=NC=C4)C=CC=3C(OC)=C(O"
            "C)C=2C=C1"
        ),
    }

    for lname in ligand_smiles:
        if not ligand_db.has_entry(key=lname):
            # Build polymer.
            molecule = stk.BuildingBlock(smiles=ligand_smiles[lname])
            molecule = stko.ETKDG().optimize(molecule)
            get_lowest_energy_xtb_conformer(
                molecule=molecule,
                name=lname,
                charge=0,
                calc_dir=calculation_dir,
                ligand_db=ligand_db,
            )

    targets = (
        ("sla_0", "sl1_0"),
        ("slb_0", "sl1_0"),
        ("slc_0", "sl1_0"),
        ("sld_0", "sl1_0"),
        ("sla_0", "sl2_0"),
        ("slb_0", "sl2_0"),
        ("slc_0", "sl2_0"),
        ("sld_0", "sl2_0"),
        ("lab_0", "la_0"),
        ("lab_0", "lb_0"),
        ("lab_0", "lc_0"),
        ("lab_0", "ld_0"),
        ("m2h_0", "m4q_0"),
        ("m2h_0", "m4p_0"),
        ("e16_0", "e10_0"),
        ("e16_0", "e17_0"),
        ("e10_0", "e17_0"),
        ("e11_0", "e10_0"),
        ("e16_0", "e14_0"),
        ("e18_0", "e14_0"),
        ("e18_0", "e10_0"),
        ("e12_0", "e10_0"),
        ("e11_0", "e14_0"),
        ("e12_0", "e14_0"),
        ("e11_0", "e13_0"),
        ("e12_0", "e13_0"),
        ("e13_0", "e14_0"),
        ("e11_0", "e12_0"),
    )

    for ligand1, ligand2 in targets:
        key = f"{ligand1}_{ligand2}"

        if pair_db.has_property_entry(key):
            continue

        logging.info("analysing %s and %s", ligand1, ligand2)
        analyse_ligand_pair(
            ligand1=ligand1,
            ligand2=ligand2,
            key=key,
            ligand_db=ligand_db,
            pair_db=pair_db,
            calculation_dir=calculation_dir,
        )

    plot_targets_sets = {
        "expt": (
            ("e16_0", "e10_0"),
            ("e16_0", "e17_0"),
            ("e10_0", "e17_0"),
            ("e11_0", "e10_0"),
            ("e16_0", "e14_0"),
            ("e18_0", "e14_0"),
            ("e18_0", "e10_0"),
            ("e12_0", "e10_0"),
            ("e11_0", "e14_0"),
            ("e12_0", "e14_0"),
            ("e11_0", "e13_0"),
            ("e12_0", "e13_0"),
            ("e13_0", "e14_0"),
            ("e11_0", "e12_0"),
        ),
        "2024": (
            ("sla_0", "sl1_0"),
            ("slb_0", "sl1_0"),
            ("slc_0", "sl1_0"),
            ("sld_0", "sl1_0"),
            ("sla_0", "sl2_0"),
            ("slb_0", "sl2_0"),
            ("slc_0", "sl2_0"),
            ("sld_0", "sl2_0"),
        ),
        "het": (
            ("lab_0", "la_0"),
            ("lab_0", "lb_0"),
            ("lab_0", "lc_0"),
            ("lab_0", "ld_0"),
            ("m2h_0", "m4q_0"),
            ("m2h_0", "m4p_0"),
        ),
    }
    for pts in plot_targets_sets:
        plot_targets = plot_targets_sets[pts]
        if pts in ("het",):
            unsymmetric_plot(
                pts=pts,
                plot_targets=plot_targets,
                figures_dir=figures_dir,
                pair_db=pair_db,
            )
        else:
            symmetric_plot(
                pts=pts,
                plot_targets=plot_targets,
                figures_dir=figures_dir,
                pair_db=pair_db,
            )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    main()
