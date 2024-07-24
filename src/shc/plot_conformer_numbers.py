"""Script to build the ligand in this project."""

import json
import logging
import os

import plotting
from definitions import EnvVariables


def main() -> None:
    """Run script."""
    raise SystemExit("convert this to taking a db as an arg to show")
    _wd = liga_path()
    _cd = calc_path()

    res_file = os.path.join(_wd, "all_ligand_res.json")
    figure_prefix = "etkdg"

    with open(res_file) as f:
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
        dihedral_cutoff=EnvVariables.dihedral_cutoff,
        strain_cutoff=EnvVariables.strain_cutoff,
        low_energy_values=low_energy_values,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    main()
