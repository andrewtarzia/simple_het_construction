working directory: /data/atarzia/projects/simple_het
env_set.py sets environment, incl. directories for output

build_ligand.py
    build ligand precusor for cage construction -- includes optimisation
    generates ligands lowest energy conformer

build_cage.py
    will build 4 cage symmetries -- includes optimisation

utilities.py
    defines utilities for construction
    same as unsymm basically.

optimisation.py
    defines optimisation sequence.
    same as unsymm.
    actually, more similar to cubism, with longer MD.

analyse_cages.py
    for all desired systems:
        calculates total energies
        calculates metal-atom order parameter
        calculates ligand strain energy
