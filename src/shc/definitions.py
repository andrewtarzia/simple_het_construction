"""Define environment variables."""

import pathlib


class EnvVariables:
    """Define the variables for working environment."""

    dihedral_cutoff = 30
    strain_cutoff = 5
    cs1_dihedral_cutoff = 20


class Study1EnvVariables:
    """Define the variables for working environment."""

    dihedral_cutoff = 10
    strain_cutoff = 5

    project_path = pathlib.Path("/home/atarzia/workingspace/simple_het/")
    liga_path = project_path / "liga"
    figu_path = project_path / "figures"
    cage_path = project_path / "cages"
    xtal_path = project_path / "xtals"
    calc_path = project_path / "calculations"
    gulp_path = pathlib.Path("/home/atarzia/software/gulp-6.1/Src/gulp")
    xtb_path = pathlib.Path("/home/atarzia/miniconda3/envs/simple_het/bin/xtb")
