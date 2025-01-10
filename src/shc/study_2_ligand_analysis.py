"""Script to build the ligand in this project."""

import itertools as it
import logging
import pathlib
import time

import atomlite
import numpy as np

from shc.definitions import EnvVariables, res_str
from shc.matching_functions import mismatch_test
from shc.utilities import to_atomlite_molecule


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


def analyse_ligand_pair(  # noqa: PLR0915
    ligand1_entry: atomlite.Entry,
    ligand2_entry: atomlite.Entry,
    pair_key: str,
    pair_db_path: atomlite.Database,
) -> None:
    """Analyse a pair of ligands."""
    pair_db = atomlite.Database(pair_db_path)
    ligand1_confs = ligand1_entry.properties["conf_data"]
    ligand2_confs = ligand2_entry.properties["conf_data"]

    st = time.time()
    num_pairs = 0
    num_pairs_passed = 0
    num_failed_strain = 0
    num_failed_torsion = 0
    cid_names = []
    state1_scores = []
    state2_scores = []
    ctimes = []
    all_scores = []
    best_residual = float("inf")
    best_pair = None
    # Iterate over the product of all conformers.
    for cid_1, cid_2 in it.product(ligand1_confs, ligand2_confs):
        cid_name = f"{cid_1}-{cid_2}"

        num_pairs += 1
        # Check strain.
        strain1 = (
            ligand1_confs[cid_1]["UFFEnergy;kj/mol"]
            - ligand1_entry.properties["min_energy;kj/mol"]
        )
        strain2 = (
            ligand2_confs[cid_2]["UFFEnergy;kj/mol"]
            - ligand2_entry.properties["min_energy;kj/mol"]
        )
        if (
            strain1 > EnvVariables.strain_cutoff
            or strain2 > EnvVariables.strain_cutoff
        ):
            num_failed_strain += 1
            continue

        # Check torsion.
        torsion1 = abs(ligand1_confs[cid_1]["NCCN_dihedral"])
        torsion2 = abs(ligand2_confs[cid_2]["NCCN_dihedral"])
        if (
            torsion1 > EnvVariables.dihedral_cutoff
            or torsion2 > EnvVariables.dihedral_cutoff
        ):
            num_failed_torsion += 1
            continue

        # Calculate geom score for both sides together.
        c_dict1 = ligand1_confs[cid_1]
        c_dict2 = ligand2_confs[cid_2]

        # Calculate final geometrical properties.
        stc = time.time()
        pair_results = mismatch_test(
            c_dict1=c_dict1,
            c_dict2=c_dict2,
            k_angle=EnvVariables.k_angle,
            k_bond=EnvVariables.k_bond,
        )
        etc = time.time()

        cid_names.append(cid_name)
        ctimes.append(float(etc - stc))
        state1_scores.append(float(pair_results.state_1_result))
        state2_scores.append(float(pair_results.state_2_result))
        all_scores.append(
            min(
                float(pair_results.state_1_result),
                float(pair_results.state_2_result),
            )
        )
        num_pairs_passed += 1
        # for idx in range(len(pair_results.state_1_parameters)):
        #     pair_db.set_property(
        # key=pair_key,# noqa: ERA001
        # path=f"{prefix_path}.state_1_parameter.{idx}",  # noqa: ERA001
        # property=float(pair_results.state_1_parameters[idx]),# noqa: ERA001
        # commit=False,# noqa: ERA001
        if float(pair_results.state_1_result) < best_residual:
            best_state = 1
            best_pair = pair_results
            best_residual = min(
                (
                    float(pair_results.state_1_result),
                    float(pair_results.state_2_result),
                )
            )
            best_s1_cid1 = cid_1
            best_s1_cid2 = cid_2

        elif float(pair_results.state_2_result) < best_residual:
            best_state = 2
            best_pair = pair_results
            best_residual = min(
                (
                    float(pair_results.state_1_result),
                    float(pair_results.state_2_result),
                )
            )
            best_s2_cid1 = cid_1
            best_s2_cid2 = cid_2

    ft = time.time()

    if num_pairs_passed == 0:
        logging.warning("make this smarter, need to add property entry logic")

    else:
        entry = atomlite.Entry(
            key=pair_key,
            molecule=to_atomlite_molecule(best_pair, best_state),
            properties={
                "cidnames": cid_names,
                "ctime/s": ctimes,
                "state_1_residuals": state1_scores,
                "state_2_residuals": state2_scores,
                "ligand1_key": ligand1_entry.key,
                "ligand2_key": ligand2_entry.key,
                "time_taken/s": float(ft - st),
                "num_pairs": num_pairs,
                "strain_cutoff": EnvVariables.strain_cutoff,
                "dihedral_cutoff": EnvVariables.dihedral_cutoff,
                "k_angle": EnvVariables.k_angle,
                "k_bond": EnvVariables.k_bond,
                "rmsd_threshold": EnvVariables.rmsd_threshold,
                "num_pairs_passed": num_pairs_passed,
                "num_failed_strain": num_failed_strain,
                "num_failed_torsion": num_failed_torsion,
                "mean_all": np.mean(all_scores),
                "min_all": np.min(all_scores),
                "mean_s1": np.mean(state1_scores),
                "min_s1": np.min(state1_scores),
                "mean_s2": np.mean(state1_scores),
                "min_s2": np.min(state2_scores),
                "best_s1_cid1": best_s1_cid1,
                "best_s1_cid2": best_s1_cid2,
                "best_s2_cid1": best_s2_cid1,
                "best_s2_cid2": best_s2_cid2,
            },
        )
        pair_db.add_entries(entry)


def reanalyse_ligand_pair(
    pair_key: str,
    pair_db_path: atomlite.Database,
) -> None:
    """Analyse a pair of ligands."""
    pair_entry = atomlite.Database(pair_db_path).get_entry(pair_key)

    target_properties = [
        "cidnames",
        "ctime/s",
        "state_1_residuals",
        "state_2_residuals",
        "ligand1_key",
        "ligand2_key",
        "time_taken/s",
        "num_pairs",
        "strain_cutoff",
        "dihedral_cutoff",
        "k_angle",
        "k_bond",
        "rmsd_threshold",
        "num_pairs_passed",
        "num_failed_strain",
        "num_failed_torsion",
        "mean_all",
        "min_all",
        "mean_s1",
        "min_s1",
        "mean_s2",
        "min_s2",
        "best_s1_cid1",
        "best_s1_cid2",
        "best_s2_cid1",
        "best_s2_cid2",
    ]
    missing_properties = [
        i for i in target_properties if i not in pair_entry.properties
    ]
    if len(missing_properties) == 0:
        return

    # Check if it is missing anything needed to rerun.
    key_features = (
        "cidnames",
        "state_1_residuals",
        "state_2_residuals",
        "time_taken/s",
        "ctime/s",
        "num_pairs",
        "num_pairs"
        "num_pairs_passed"
        "num_failed_strain"
        "num_failed_torsion",
    )
    if len([i for i in missing_properties if i in key_features]) > 0:
        msg = (
            f"missing a necessary feature, I deleted this entry ({pair_key})"
            f" from {pair_db_path}, please rerun"
        )
        atomlite.Database(pair_db_path).remove_entries(kets=pair_key)
        raise RuntimeError(msg)

    # Update those that are possible.
    properties = pair_entry.properties
    all_scores = []
    state1_scores = []
    state2_scores = []
    best_s1_cid1 = None
    best_s1_cid2 = None
    best_s2_cid1 = None
    best_s2_cid2 = None
    best_s1_residual = float("inf")
    best_s2_residual = float("inf")
    for cid_pair_idx, cid_name in enumerate(properties["cidnames"]):
        cid_1, cid_2 = cid_name.split("-")
        s1_score = float(properties["state_1_residuals"][cid_pair_idx])
        s2_score = float(properties["state_2_residuals"][cid_pair_idx])
        state1_scores.append(s1_score)
        state2_scores.append(s2_score)
        all_scores.append(min(s1_score, s2_score))

        if s1_score < best_s1_residual:
            best_s1_cid1 = cid_1
            best_s1_cid2 = cid_2
            best_s1_residual = s1_score

        elif s2_score < best_s2_residual:
            best_s2_cid1 = cid_1
            best_s2_cid2 = cid_2
            best_s2_residual = s2_score

    new_properties = {
        "mean_all": np.mean(all_scores),
        "min_all": np.min(all_scores),
        "mean_s1": np.mean(state1_scores),
        "min_s1": np.min(state1_scores),
        "mean_s2": np.mean(state1_scores),
        "min_s2": np.min(state2_scores),
        "best_s1_cid1": best_s1_cid1,
        "best_s1_cid2": best_s1_cid2,
        "best_s2_cid1": best_s2_cid1,
        "best_s2_cid2": best_s2_cid2,
    }

    atomlite.Database(pair_db_path).update_properties(
        atomlite.PropertyEntry(key=pair_key, properties=new_properties)
    )


    )


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
        )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    main()
