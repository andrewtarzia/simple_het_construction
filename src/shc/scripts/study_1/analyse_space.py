"""Script to analyse ligand space."""

import json
import logging
import pathlib

import matplotlib.pyplot as plt

from shc.definitions import Study1EnvVariables


def plot_conformer_props(
    structure_results: dict,
    dihedral_cutoff: float,
    strain_cutoff: float,
    low_energy_values: dict,
    output_path: pathlib.Path,
) -> None:
    """Plot conformer properties."""
    logging.info("plotting: plot_conformer_props")

    fig, ax = plt.subplots(figsize=(8, 5))

    for ligand, sres in structure_results.items():
        if "e" in ligand:
            continue
        if "ll" in ligand:
            continue
        if "ls" in ligand:
            continue
        original_number = 500
        after_rmsd = len(sres)

        low_energy = low_energy_values[ligand][1]

        within_strain = {}
        for cid in sres:
            strain = sres[cid]["UFFEnergy;kj/mol"] - low_energy
            if strain <= strain_cutoff:
                within_strain[cid] = sres[cid]
        after_strain = len(within_strain)

        within_torsion = {}
        for cid, strain_count in within_strain.items():
            dihedral = within_strain[cid]["NCCN_dihedral"]
            if abs(dihedral) <= dihedral_cutoff:
                within_torsion[cid] = strain_count

        after_torsion = len(within_torsion)
        logging.info(
            "%s: %s, %s, %s, %s",
            ligand,
            original_number,
            after_rmsd,
            after_strain,
            after_torsion,
        )
        if ligand in ("l1", "l2", "l3"):
            label = "1$^{\\mathrm{X}}$" if ligand == "l1" else None
            c = "#26547C"
        else:
            c = "#FE6D73"
            label = "2$^{\\mathrm{X}}$" if ligand == "la" else None
        ax.plot(
            [0, 1, 2, 3],
            [original_number, after_rmsd, after_strain, after_torsion],
            lw=2,
            c=c,
            marker="o",
            markersize=8,
            label=label,
        )
    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel("screening stage", fontsize=16)
    ax.set_ylabel("number conformers", fontsize=16)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(["initial", "RMSD", "strain", "torsion"])
    ax.legend(fontsize=16)

    fig.tight_layout()
    fig.savefig(output_path, dpi=720, bbox_inches="tight")
    plt.close()


def main() -> None:
    """Run script."""
    working_dir = pathlib.Path("/home/atarzia/workingspace/cpl/study_1")
    figures_dir = pathlib.Path(
        "/home/atarzia/workingspace/cpl/figures/study_1"
    )

    res_file = working_dir / "all_ligand_res.json"
    figure_prefix = "etkdg"

    with res_file.open("r") as f:
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

    plot_conformer_props(
        structure_results=structure_results,
        output_path=figures_dir / f"{figure_prefix}_conformer_properties.png",
        dihedral_cutoff=Study1EnvVariables.dihedral_cutoff,
        strain_cutoff=Study1EnvVariables.strain_cutoff,
        low_energy_values=low_energy_values,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    main()
