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


def plot_all_data(  # noqa: PLR0915
    ligand_db: atomlite.Database,
    figures_dir: pathlib.Path,
) -> None:
    """Plot all ligand data."""
    fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(16, 10))
    (ax1, ax2), (ax3, ax4) = axs

    all_x3s = []
    all_y3s = []
    all_x4s = {}
    all_xs = []
    all_ys = []
    all_nrotatables = []
    all_shuhei_length = []
    zero_counts = []
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

        all_x3s.append(num_stable)
        all_y3s.append(np.max(delta_angles) - np.min(delta_angles))

        distances = [conf_data[i]["NN_distance"] for i in low_energy_states]
        range_distances = np.max(distances) - np.min(distances)
        angles = [
            sum(conf_data[i]["NN_BCN_angles"]) for i in low_energy_states
        ]
        range_angles = np.max(angles) - np.min(angles)
        all_xs.append(range_distances)
        all_ys.append(range_angles)

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

        untwisted_states = [
            i
            for i in low_energy_states
            if abs(conf_data[i]["NCCN_dihedral"])
            <= EnvVariables.dihedral_cutoff
        ]

        if range_distances < 1 and range_angles < 10:  # noqa: PLR2004
            cong_flex = 0
        elif range_distances < 2 and range_angles < 20:  # noqa: PLR2004
            cong_flex = 1
        elif range_distances < 5 and range_angles < 50:  # noqa: PLR2004
            cong_flex = 2
        elif range_distances < 10 and range_angles < 100:  # noqa: PLR2004
            cong_flex = 3
        else:
            cong_flex = 4

        if cong_flex not in all_x4s:
            all_x4s[cong_flex] = []
        all_x4s[cong_flex].append(len(untwisted_states))
        if len(untwisted_states) == 0:
            zero_counts.append(entry.key)

    norm = colors.LogNorm()

    cm = colors.LinearSegmentedColormap.from_list(
        "test", [(1.0, 1.0, 1.0), (0 / 255, 89 / 255, 147 / 255)], N=10
    )
    hist = ax1.hist2d(
        all_xs,
        all_ys,
        bins=[40, 40],
        range=[
            (int(min(all_xs) - 1), int(max(all_xs) + 1)),
            (int(min(all_ys) - 1), int(max(all_ys) + 1)),
        ],
        density=False,
        norm=norm,
        cmap=cm,
    )

    ax1.tick_params(axis="both", which="major", labelsize=16)
    ax1.set_ylabel(r"range (sum binder angles) [$^\circ$]", fontsize=16)
    ax1.set_xlabel(r"range (N-N distance) [$\mathrm{\AA}$]", fontsize=16)
    ax1.set_xlim(0, None)
    ax1.set_ylim(0, None)
    ax1.set_title("flexibility", fontsize=16)
    ax1.plot((0, 1, 1), (10, 10, 0), c="k", ls="--")
    ax1.plot((0, 2, 2), (20, 20, 0), c="k", ls="--")
    ax1.plot((0, 5, 5), (50, 50, 0), c="k", ls="--")
    ax1.plot((0, 10, 10), (100, 100, 0), c="k", ls="--")
    cbar = fig.colorbar(hist[3], ax=ax1)
    cbar.ax.set_ylabel("count", fontsize=16)
    cbar.ax.tick_params(labelsize=16)

    cm = colors.LinearSegmentedColormap.from_list(
        "test", [(1.0, 1.0, 1.0), (220 / 255, 96 / 255, 0 / 255)], N=10
    )
    hist = ax2.hist2d(
        all_shuhei_length,
        all_nrotatables,
        bins=[40, 40],
        range=[
            (int(min(all_shuhei_length) - 1), int(max(all_shuhei_length) + 1)),
            (int(min(all_nrotatables) - 1), int(max(all_nrotatables) + 1)),
        ],
        density=False,
        norm=norm,
        cmap=cm,
    )

    ax2.tick_params(axis="both", which="major", labelsize=16)
    ax2.set_ylabel("num. rotatable bonds", fontsize=16)
    ax2.set_xlabel("graph length", fontsize=16)
    ax2.set_xlim(0, None)
    ax2.set_ylim(0, None)
    ax2.set_title("furukawa flex", fontsize=16)
    cbar = fig.colorbar(hist[3], ax=ax2)
    cbar.ax.set_ylabel("count", fontsize=16)
    cbar.ax.tick_params(labelsize=16)

    cm = colors.LinearSegmentedColormap.from_list(
        "test", [(1.0, 1.0, 1.0), (8 / 255, 127 / 255, 20 / 255)], N=10
    )
    hist = ax3.hist2d(
        all_x3s,
        all_y3s,
        bins=[40, 40],
        range=[(0, 500), (0, int(max(all_y3s) + 1))],
        density=False,
        norm=norm,
        cmap=cm,
    )
    ax3.tick_params(axis="both", which="major", labelsize=16)
    ax3.set_xlabel("num. stable conformers", fontsize=16)
    ax3.set_ylabel(r"range |$\Delta$ binder angles| [$^\circ$]", fontsize=16)
    ax3.set_xlim(0, None)
    ax3.set_ylim(0, None)
    ax3.set_title("asymmetry", fontsize=16)
    cbar = fig.colorbar(hist[3], ax=ax3)
    cbar.ax.set_ylabel("count", fontsize=16)
    cbar.ax.tick_params(labelsize=16)

    for x in all_x4s:
        ax4.boxplot(x=all_x4s[x], positions=[x], widths=[0.8])
    ax4.tick_params(axis="both", which="major", labelsize=16)
    ax4.set_xlabel("conformer flex region", fontsize=16)
    ax4.set_ylabel("num. passed", fontsize=16)
    ax4.set_ylim(0, None)
    ax4.set_title(
        f"conformer counts ({len(zero_counts)} with 0 passed)", fontsize=16
    )

    fig.tight_layout()
    fig.savefig(
        figures_dir / "all_ligand_data.png",
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

    plot_all_data(ligand_db=deduped_db, figures_dir=figures_dir)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    main()
