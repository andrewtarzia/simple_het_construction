# simple_het_construction
Python code for the construction and analysis of heteroleptic cages

DOI for publication: **AWAITING**.

The library is built off of [`stk`](https://stk.readthedocs.io/en/stable/)

# Installation

I recommend installing the library with the following instructions and setting the paths in `env_set.py` appropriately. *E.g.*, the working directory for me is `/home/atarzia/projects/simple_het/`

The code can be installed following these steps:

1. clone the `submission` branch of `simple_het_construction` from [here](https://github.com/andrewtarzia/simple_het_construction)

2. Create a `conda` or `mamba` environment:
```
conda env create -f environment.yml
```

3. Update directories in `env_set.py`

For cage and ligand optimisation, this library uses:

`Gulp (6.1)`: Follow the instructions to download and install [GULP](https://gulp.curtin.edu.au/gulp/help/manuals.cfm)

`xTB (6.5.0=h9d67668_0)`: Installed using `conda` or `mamba` through `environment.yml`

# Usage for ligand-based modelling

`run_ligand_analysis.py`:
    Runs the ligand-based analysis (conformer generation and pairing).

`analyse_csd_survey.py`:
    This is a helper script for analysing the survery of Pd centres in the Cambridge Structural Database. The methods for extracting the data are not provided.

`plot_conformer_numbers.py`:
    Plots the number of conformers for each ligand at each stage of screening before performing ligand-based analysis.

`run_pdn_distance_tests.py`:
    Reruns some of the ligand-based analysis with varying Pd-N distances to see impact of this variable.

# Usage for cage modelling

`extract_m30l60.py`:
    Extracts the topology information from a given structure.

`build_ligands.py`:
    Build ligands for cage construction. Generates ligands lowest energy conformer and an appropriate conformer for cage construction.

`build_cages.py`:
    Builds and optimises homoleptic and heteroleptic cages. List of structures to build are defined in the script.

`analyse_cages.py`:
    Obtains structural and energetic analysis for all cages. Produces plots.

# Modules

`utilities.py`:
    Defines utilities for construction and analysis.

`optimisation.py`:
    Defines optimisation sequences.

`topologies.py`:
    Defines the new topology graph for M30L60.

`plotting.py`:
    Utilities and functions for plotting.

`pywindow_module.py`:
    Defines a class for using `pyWindow`.
