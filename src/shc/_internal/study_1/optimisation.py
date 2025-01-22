"""Module for optimisation functions."""

import logging
import pathlib

import stk
import stko

from shc.definitions import Study1EnvVariables


def optimisation_sequence(
    mol: stk.Molecule,
    name: str,
    charge: int,
    calc_dir: pathlib.Path,
) -> stk.Molecule:
    gulp1_output = calc_dir / f"{name}_gulp1.mol"
    gulp2_output = calc_dir / f"{name}_gulp2.mol"
    gulpmd_output = calc_dir / f"{name}_gulpmd.mol"
    xtbopt_output = calc_dir / f"{name}_xtb.mol"
    xtbsolvopt_output = calc_dir / f"{name}_xtb_dmso.mol"

    if not gulp1_output.exists():
        output_dir = calc_dir / f"{name}_gulp1"

        gulp_opt = stko.GulpUFFOptimizer(
            gulp_path=Study1EnvVariables.gulp_path,
            maxcyc=1000,
            metal_FF={46: "Pd4+2"},
            metal_ligand_bond_order="",
            output_dir=output_dir,
            conjugate_gradient=True,
        )
        gulp_opt.assign_FF(mol)
        gulp1_mol = gulp_opt.optimize(mol=mol)
        gulp1_mol.write(gulp1_output)
    else:
        gulp1_mol = mol.with_structure_from_file(gulp1_output)

    if not gulp2_output.exists():
        output_dir = calc_dir / f"{name}_gulp2"

        gulp_opt = stko.GulpUFFOptimizer(
            gulp_path=Study1EnvVariables.gulp_path(),
            maxcyc=1000,
            metal_FF={46: "Pd4+2"},
            metal_ligand_bond_order="",
            output_dir=output_dir,
            conjugate_gradient=False,
        )
        gulp_opt.assign_FF(gulp1_mol)
        gulp2_mol = gulp_opt.optimize(mol=gulp1_mol)
        gulp2_mol.write(gulp2_output)
    else:
        gulp2_mol = mol.with_structure_from_file(gulp2_output)

    if not gulpmd_output.exists():
        gulp_md = stko.GulpUFFMDOptimizer(
            gulp_path=Study1EnvVariables.gulp_path(),
            metal_FF={46: "Pd4+2"},
            metal_ligand_bond_order="",
            output_dir=calc_dir / f"{name}_gulpmde",
            integrator="leapfrog verlet",
            ensemble="nvt",
            temperature=1000,
            timestep=0.25,
            equilbration=0.5,
            production=0.5,
            N_conformers=2,
            opt_conformers=False,
            save_conformers=False,
        )
        gulp_md.assign_FF(gulp2_mol)
        gulpmd_mol = gulp_md.optimize(mol=gulp2_mol)

        gulp_md = stko.GulpUFFMDOptimizer(
            gulp_path=Study1EnvVariables.gulp_path(),
            metal_FF={46: "Pd4+2"},
            metal_ligand_bond_order="",
            output_dir=calc_dir / f"{name}_gulpmd",
            integrator="leapfrog verlet",
            ensemble="nvt",
            temperature=1000,
            timestep=0.75,
            equilbration=0.5,
            production=200.0,
            N_conformers=40,
            opt_conformers=True,
            save_conformers=False,
        )
        gulp_md.assign_FF(gulpmd_mol)
        gulpmd_mol = gulp_md.optimize(mol=gulpmd_mol)
        gulpmd_mol.write(gulpmd_output)
    else:
        gulpmd_mol = mol.with_structure_from_file(gulpmd_output)

    if not xtbopt_output.exists():
        output_dir = calc_dir / f"{name}_xtbopt"

        xtb_opt = stko.XTB(
            xtb_path=Study1EnvVariables.xtb_path(),
            output_dir=output_dir,
            gfn_version=2,
            num_cores=6,
            charge=charge,
            opt_level="normal",
            num_unpaired_electrons=0,
            max_runs=1,
            calculate_hessian=False,
            unlimited_memory=True,
            solvent=None,
        )
        xtbopt_mol = xtb_opt.optimize(mol=gulpmd_mol)
        xtbopt_mol.write(xtbopt_output)
    else:
        xtbopt_mol = mol.with_structure_from_file(xtbopt_output)

    if not xtbsolvopt_output.exists():
        output_dir = calc_dir / f"{name}_xtbsolvopt"

        xtb_opt = stko.XTB(
            xtb_path=Study1EnvVariables.xtb_path,
            output_dir=output_dir,
            gfn_version=2,
            num_cores=6,
            charge=charge,
            opt_level="normal",
            num_unpaired_electrons=0,
            max_runs=1,
            calculate_hessian=False,
            unlimited_memory=True,
            solvent_model="alpb",
            solvent="dmso",
            solvent_grid="verytight",
        )
        xtbsolvopt_mol = xtb_opt.optimize(mol=xtbopt_mol)
        xtbsolvopt_mol.write(xtbsolvopt_output)
    else:
        logging.info("loading %s", xtbsolvopt_output)
        xtbsolvopt_mol = mol.with_structure_from_file(xtbsolvopt_output)

    return mol.with_structure_from_file(xtbsolvopt_output)
