"""Module for optimisation functions."""

import logging
import os

import stko


def optimisation_sequence(mol, name, charge, calc_dir):
    gulp1_output = str(calc_dir / f"{name}_gulp1.mol")
    gulp2_output = str(calc_dir / f"{name}_gulp2.mol")
    gulpmd_output = str(calc_dir / f"{name}_gulpmd.mol")
    xtbopt_output = str(calc_dir / f"{name}_xtb.mol")
    xtbsolvopt_output = str(calc_dir / f"{name}_xtb_dmso.mol")

    if not os.path.exists(gulp1_output):
        output_dir = os.path.join(calc_dir, f"{name}_gulp1")
        CG = True
        logging.info(f"UFF4MOF optimisation 1 of {name} CG: {CG}")
        gulp_opt = stko.GulpUFFOptimizer(
            gulp_path=env_set.gulp_path(),
            maxcyc=1000,
            metal_FF={46: "Pd4+2"},
            metal_ligand_bond_order="",
            output_dir=output_dir,
            conjugate_gradient=CG,
        )
        gulp_opt.assign_FF(mol)
        gulp1_mol = gulp_opt.optimize(mol=mol)
        gulp1_mol.write(gulp1_output)
    else:
        logging.info(f"loading {gulp1_output}")
        gulp1_mol = mol.with_structure_from_file(gulp1_output)

    if not os.path.exists(gulp2_output):
        output_dir = os.path.join(calc_dir, f"{name}_gulp2")
        CG = False
        logging.info(f"UFF4MOF optimisation 2 of {name} CG: {CG}")
        gulp_opt = stko.GulpUFFOptimizer(
            gulp_path=env_set.gulp_path(),
            maxcyc=1000,
            metal_FF={46: "Pd4+2"},
            metal_ligand_bond_order="",
            output_dir=output_dir,
            conjugate_gradient=CG,
        )
        gulp_opt.assign_FF(gulp1_mol)
        gulp2_mol = gulp_opt.optimize(mol=gulp1_mol)
        gulp2_mol.write(gulp2_output)
    else:
        logging.info(f"loading {gulp2_output}")
        gulp2_mol = mol.with_structure_from_file(gulp2_output)

    if not os.path.exists(gulpmd_output):
        logging.info(f"UFF4MOF equilib MD of {name}")
        gulp_MD = stko.GulpUFFMDOptimizer(
            gulp_path=env_set.gulp_path(),
            metal_FF={46: "Pd4+2"},
            metal_ligand_bond_order="",
            output_dir=os.path.join(calc_dir, f"{name}_gulpmde"),
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
        gulp_MD.assign_FF(gulp2_mol)
        gulpmd_mol = gulp_MD.optimize(mol=gulp2_mol)

        logging.info(f"UFF4MOF production MD of {name}")
        gulp_MD = stko.GulpUFFMDOptimizer(
            gulp_path=env_set.gulp_path(),
            metal_FF={46: "Pd4+2"},
            metal_ligand_bond_order="",
            output_dir=os.path.join(calc_dir, f"{name}_gulpmd"),
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
        gulp_MD.assign_FF(gulpmd_mol)
        gulpmd_mol = gulp_MD.optimize(mol=gulpmd_mol)
        gulpmd_mol.write(gulpmd_output)
    else:
        logging.info(f"loading {gulpmd_output}")
        gulpmd_mol = mol.with_structure_from_file(gulpmd_output)

    if not os.path.exists(xtbopt_output):
        output_dir = os.path.join(calc_dir, f"{name}_xtbopt")
        logging.info(f"xtb optimisation of {name}")
        xtb_opt = stko.XTB(
            xtb_path=env_set.xtb_path(),
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
        logging.info(f"loading {xtbopt_output}")
        xtbopt_mol = mol.with_structure_from_file(xtbopt_output)

    if not os.path.exists(xtbsolvopt_output):
        output_dir = os.path.join(calc_dir, f"{name}_xtbsolvopt")
        logging.info(f"solvated xtb optimisation of {name}")
        xtb_opt = stko.XTB(
            xtb_path=env_set.xtb_path(),
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
        logging.info(f"loading {xtbsolvopt_output}")
        xtbsolvopt_mol = mol.with_structure_from_file(xtbsolvopt_output)

    final_mol = mol.with_structure_from_file(xtbsolvopt_output)
    return final_mol
