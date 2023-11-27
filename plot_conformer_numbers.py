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
import json

from env_set import liga_path, calc_path
import plotting


def main():
    if not len(sys.argv) == 1:
        logging.info(f"Usage: {__file__}\n" "   Expected 0 arguments:")
        sys.exit()
    else:
        pass

    _wd = liga_path()
    _cd = calc_path()

    dihedral_cutoff = 10
    strain_cutoff = 5

    res_file = os.path.join(_wd, "all_ligand_res.json")
    figure_prefix = "etkdg"

    with open(res_file, "r") as f:
        structure_results = json.load(f)

    # Define minimum energies for all ligands.
    low_energy_values = {}
    for ligand in structure_results:
        sres = structure_results[ligand]
        min_energy = 1e24
        min_e_cid = 0
        for cid in sres:
            energy = sres[cid]["UFFEnergy;kj/mol"]
            if energy < min_energy:
                min_energy = energy
                min_e_cid = cid
        low_energy_values[ligand] = (min_e_cid, min_energy)

    plotting.plot_conformer_props(
        structure_results=structure_results,
        outname=f"{figure_prefix}_conformer_properties.png",
        dihedral_cutoff=dihedral_cutoff,
        strain_cutoff=strain_cutoff,
        low_energy_values=low_energy_values,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    main()
