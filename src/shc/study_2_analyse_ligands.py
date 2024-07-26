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
from rdkit.Chem import rdMolDescriptors, rdmolops, rdMolTransforms


def extract_torsions(
    molecule: stk.Molecule,
    smarts: str,
    expected_num_atoms: int,
    scanned_ids: tuple[int, int, int, int],
    expected_num_torsions: int,
) -> tuple[float, float]:
    """Extract two torsions from a molecule."""
    rdkit_molecule = molecule.to_rdkit_mol()
    matches = rdkit_molecule.GetSubstructMatches(
        query=rdkit.MolFromSmarts(smarts),
    )

    torsions = []
    for match in matches:
        if len(match) != expected_num_atoms:
            msg = f"{len(match)} not as expected ({expected_num_atoms})"
            raise RuntimeError(msg)
        torsions.append(
            abs(
                rdMolTransforms.GetDihedralDeg(
                    rdkit_molecule.GetConformer(0),
                    match[scanned_ids[0]],
                    match[scanned_ids[1]],
                    match[scanned_ids[2]],
                    match[scanned_ids[3]],
                )
            )
        )
    if len(torsions) != expected_num_torsions:
        msg = f"{len(torsions)} found, not {expected_num_torsions}!"
        raise RuntimeError(msg)
    return tuple(torsions)


def get_amide_torsions(molecule: stk.Molecule) -> tuple[float, float]:
    """Get the centre alkene torsion from COOH."""
    smarts = "[#6X3H0]~[#6X3H1]~[#6X3H0]~[#6X3H0](=[#8])~[#7]"
    expected_num_atoms = 6
    scanned_ids = (1, 2, 3, 5)

    return extract_torsions(
        molecule=molecule,
        smarts=smarts,
        expected_num_atoms=expected_num_atoms,
        scanned_ids=scanned_ids,
        expected_num_torsions=1,
    )


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
    plot_flexes(ligand_db=deduped_db, figures_dir=figures_dir)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    main()
