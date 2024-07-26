"""Script to analyse the ligands in this project."""

import logging
import pathlib

import atomlite
import matplotlib.pyplot as plt
import numpy as np
import stk
import stko
from definitions import EnvVariables
from matplotlib import colors
from rdkit.Chem import AllChem as rdkit  # noqa: N813
from rdkit.Chem import rdMolDescriptors, rdmolops


def get_num_alkynes(rdkit_mol: rdkit.Mol) -> int:
    """Get the number of alkynes in a molecule."""
    smarts = "[#6]#[#6]"
    matches = rdkit_mol.GetSubstructMatches(
        query=rdkit.MolFromSmarts(smarts),
    )
    return len(matches)


def plot_flexes(
    ligand_db: atomlite.Database,
    figures_dir: pathlib.Path,
) -> None:
    """Plot ligand flexibilities."""
    fig, axs = plt.subplots(ncols=2, figsize=(16, 5))
    ax, ax1 = axs
    all_xs = []
    all_ys = []
    all_strs = []
    all_nrotatables = []
    all_shuhei_length = []
    for entry in ligand_db.get_entries():
        conf_data = entry.properties["conf_data"]
        min_energy = min([conf_data[i]["UFFEnergy;kj/mol"] for i in conf_data])
        low_energy_states = [
            i
            for i in conf_data
            if (conf_data[i]["UFFEnergy;kj/mol"] - min_energy)
            < EnvVariables.strain_cutoff
        ]
        sigma_distances = np.std(
            [conf_data[i]["NN_distance"] for i in low_energy_states]
        )
        sigma_angles = np.std(
            [sum(conf_data[i]["NN_BCN_angles"]) for i in low_energy_states]
        )
        all_xs.append(sigma_distances)
        all_ys.append(sigma_angles)
        all_strs.append(entry.key)
        rdkit_mol = atomlite.json_to_rdkit(entry.molecule)
        rdmolops.SanitizeMol(rdkit_mol)
        num_rotatable_bonds = rdMolDescriptors.CalcNumRotatableBonds(
            rdkit_mol, rdMolDescriptors.NumRotatableBondsOptions.NonStrict
        ) + get_num_alkynes(rdkit_mol)
        all_nrotatables.append(num_rotatable_bonds)

        new_mol = stk.BuildingBlock.init_from_rdkit_mol(
            rdkit_mol,
            functional_groups=stko.functional_groups.ThreeSiteFactory(
                smarts="[#6]~[#7X2]~[#6]", bonders=(1,), deleters=()
            ),
        )
        target_atom_ids = tuple(
            next(iter(i.get_bonder_ids()))
            for i in new_mol.get_functional_groups()
        )
        dist_matrix = rdmolops.GetDistanceMatrix(rdkit_mol)
        shuhei_length = dist_matrix[target_atom_ids[0]][target_atom_ids[1]]
        all_shuhei_length.append(shuhei_length)

    norm = colors.LogNorm()

    cm = colors.LinearSegmentedColormap.from_list(
        "test", [(1.0, 1.0, 1.0), (255 / 255, 87 / 255, 51 / 255)], N=10
    )
    hist = ax.hist2d(
        all_xs,
        all_ys,
        bins=[40, 40],
        range=[(0, 5), (0, 80)],
        density=False,
        norm=norm,
        cmap=cm,
    )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_ylabel(r"$\sigma$ (sum binder angles) [deg]", fontsize=16)
    ax.set_xlabel(r"$\sigma$ (N-N distance) [AA]", fontsize=16)
    ax.set_ylim(0, None)
    ax.set_xlim(0, None)
    ax.set_title(len(all_xs), fontsize=16)
    cbar = fig.colorbar(hist[3], ax=ax)
    cbar.ax.set_ylabel("count", fontsize=16)
    cbar.ax.tick_params(labelsize=16)

    hist = ax1.hist2d(
        all_shuhei_length,
        all_nrotatables,
        bins=[40, 40],
        range=[(0, 30), (0, 12)],
        density=False,
        norm=norm,
        cmap=cm,
    )

    ax1.tick_params(axis="both", which="major", labelsize=16)
    ax1.set_ylabel("num. rotatable bonds", fontsize=16)
    ax1.set_xlabel("graph length", fontsize=16)
    ax1.set_ylim(0, None)
    ax1.set_xlim(0, None)
    ax1.set_title(len(all_shuhei_length), fontsize=16)
    cbar = fig.colorbar(hist[3], ax=ax1)
    cbar.ax.set_ylabel("count", fontsize=16)
    cbar.ax.tick_params(labelsize=16)

    fig.tight_layout()
    fig.savefig(
        figures_dir / "all_flexes.png",
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()


def plot_asymmetries(
    ligand_db: atomlite.Database,
    figures_dir: pathlib.Path,
) -> None:
    """Plot ligand flexibilities."""
    fig, ax = plt.subplots(figsize=(8, 5))

    all_xs = []
    all_ys = []
    for entry in ligand_db.get_entries():
        conf_data = entry.properties["conf_data"]
        min_energy = min([conf_data[i]["UFFEnergy;kj/mol"] for i in conf_data])
        low_energy_states = [
            i
            for i in conf_data
            if (conf_data[i]["UFFEnergy;kj/mol"] - min_energy)
            < EnvVariables.strain_cutoff
        ]

        delta_angles = [
            abs(
                conf_data[i]["NN_BCN_angles"][0]
                - conf_data[i]["NN_BCN_angles"][1]
            )
            for i in low_energy_states
        ]
        num_stable = len(low_energy_states)

        all_xs.append(num_stable)
        all_ys.append(np.mean(delta_angles))

    norm = colors.LogNorm()
    cm = colors.LinearSegmentedColormap.from_list(
        "test", [(1.0, 1.0, 1.0), (255 / 255, 87 / 255, 51 / 255)], N=10
    )
    hist = ax.hist2d(
        all_xs,
        all_ys,
        bins=[40, 40],
        range=[(0, 500), (0, int(max(all_ys) + 1))],
        density=False,
        norm=norm,
        cmap=cm,
    )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel("num stable conformers", fontsize=16)
    ax.set_ylabel(r"mean |$\Delta$ binder angles| [deg]", fontsize=16)
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)
    ax.set_title(len(all_xs), fontsize=16)
    cbar = fig.colorbar(hist[3], ax=ax)
    cbar.ax.set_ylabel("count", fontsize=16)
    cbar.ax.tick_params(labelsize=16)

    fig.tight_layout()
    fig.savefig(
        figures_dir / "all_asymmetries.png",
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()


def main() -> None:
    """Run script."""
    ligand_dir = pathlib.Path("/home/atarzia/workingspace/cpl/ligand_analysis")
    calculation_dir = pathlib.Path(
        "/home/atarzia/workingspace/cpl/calculations"
    )
    figures_dir = pathlib.Path(
        "/home/atarzia/workingspace/cpl/figures/ligand_analysis"
    )
    ligand_dir.mkdir(exist_ok=True)
    calculation_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True, parents=True)

    deduped_db = atomlite.Database(ligand_dir / "deduped_ligands.db")

    logging.info("built %s deduped ligands", deduped_db.num_entries())
    logging.info(
        "%s ligands in key database",
        atomlite.Database(ligand_dir / "keys.db").num_entries() / 3,
    )

    plot_asymmetries(ligand_db=deduped_db, figures_dir=figures_dir)
    plot_flexes(ligand_db=deduped_db, figures_dir=figures_dir)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    main()
