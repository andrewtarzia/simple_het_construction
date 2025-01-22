"""Script to build the ligand in this project."""

import argparse
import logging
import pathlib

import atomlite
import numpy as np

from .study_2_utilities import (
    analyse_ligand_pair,
    plot_states,
    plot_timings,
    reanalyse_ligand_pair,
    to_cg_chemiscope,
    to_chemiscope,
    to_csv,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run",
        action="store_true",
        help="True to run calculations of pair matching",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="True to output materials for sharing with collaborators",
    )

    return parser.parse_args()


def main() -> None:
    """Run script."""
    args = _parse_args()
    ligand_dir = pathlib.Path("/home/atarzia/workingspace/cpl/ligand_analysis")
    calculation_dir = pathlib.Path(
        "/home/atarzia/workingspace/cpl/calculations"
    )
    figures_dir = pathlib.Path(
        "/home/atarzia/workingspace/cpl/figures/screening"
    )
    ligand_dir.mkdir(exist_ok=True)
    calculation_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)
    ligand_db_path = ligand_dir / "deduped_ligands.db"
    pair_db_path = ligand_dir / "pairs.db"

    if args.run:
        logging.info("loading ligands...")
        ligand_entries = list(atomlite.Database(ligand_db_path).get_entries())

        # Shuffle the list.
        rng = np.random.default_rng(seed=267)
        rng.shuffle(ligand_entries)

        logging.info(
            "iterating, but with random choice, one day make systematic"
        )

        target_count = 500
        for _ in range(target_count):
            ligand1_entry, ligand2_entry = rng.choice(ligand_entries, size=2)

            ligand1, ligand2 = sorted((ligand1_entry.key, ligand2_entry.key))
            pair_key = f"{ligand1}_{ligand2}"

            if ligand1 == ligand2:
                continue
            if atomlite.Database(pair_db_path).has_entry(pair_key):
                reanalyse_ligand_pair(
                    pair_key=pair_key,
                    pair_db_path=pair_db_path,
                )

            if atomlite.Database(pair_db_path).has_property_entry(pair_key):
                continue

            logging.info(
                "(%s/%s) analysing %s and %s",
                _,
                target_count,
                ligand1,
                ligand2,
            )

            analyse_ligand_pair(
                ligand1_entry=ligand1_entry,
                ligand2_entry=ligand2_entry,
                pair_key=pair_key,
                pair_db_path=pair_db_path,
            )

    if args.share:
        plot_timings(pair_db_path=pair_db_path, figures_dir=figures_dir)
        plot_states(pair_db_path=pair_db_path, figures_dir=figures_dir)
        to_cg_chemiscope(
            pair_db_path=pair_db_path,
            figures_dir=figures_dir,
        )
        to_chemiscope(
            pair_db_path=pair_db_path,
            ligand_dir=ligand_dir,
            figures_dir=figures_dir,
        )
        to_csv(
            pair_db_path=pair_db_path,
            ligand_dir=ligand_dir,
            figures_dir=figures_dir,
        )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    main()
