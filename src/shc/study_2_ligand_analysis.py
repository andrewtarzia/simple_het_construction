"""Script to build the ligand in this project."""

import itertools as it
import logging
import pathlib
import time

import atomlite
import numpy as np
from definitions import EnvVariables


def vector_length():
    """Mean value of bond distance to use in candidate selection."""
    return 2.02


def get_test_1(large_c_dict, small_c_dict):
    raise SystemExit("need to make sure this is pointing the right way")
    # l_angle = (large_c_dict["NN_BCN_angles"]["NN_BCN1"] - 90) + (
    #     large_c_dict["NN_BCN_angles"]["NN_BCN2"] - 90
    # )

    # s_angle = small_c_dict["bite_angle"]
    # # Version 1
    # return s_angle / l_angle
    # # Version 2
    # return (s_angle + l_angle) / 180

    # Version 3.
    # 180 - angle, to make it the angle toward the binding interaction.
    # E.g. To become internal angle of trapezoid.
    l_angle_1 = 180 - large_c_dict["NN_BCN_angles"]["NN_BCN1"]
    l_angle_2 = 180 - large_c_dict["NN_BCN_angles"]["NN_BCN2"]

    s_angle_1 = 180 - small_c_dict["NN_BCN_angles"]["NN_BCN1"]
    s_angle_2 = 180 - small_c_dict["NN_BCN_angles"]["NN_BCN2"]

    interior_angles = l_angle_1 + l_angle_2 + s_angle_1 + s_angle_2
    return interior_angles / 360


def get_test_2(large_c_dict, small_c_dict, pdn_distance):
    raise SystemExit("need to make sure this is pointing the right way")
    sNN_dist = small_c_dict["NN_distance"]
    lNN_dist = large_c_dict["NN_distance"]
    # 180 - angle, to make it the angle toward the binding interaction.
    # E.g. To become internal angle of trapezoid.
    s_angle1 = 180 - small_c_dict["NN_BCN_angles"]["NN_BCN1"]
    s_angle2 = 180 - small_c_dict["NN_BCN_angles"]["NN_BCN2"]

    bonding_vector_length = 2 * pdn_distance
    se1 = bonding_vector_length * np.sin(np.radians(s_angle1))
    se2 = bonding_vector_length * np.sin(np.radians(s_angle2))

    ideal_dist = sNN_dist + se1 + se2
    return lNN_dist / ideal_dist


def test_N_N_lengths(large_c_dict, small_c_dict):
    large_NN_distance = large_c_dict["NN_distance"]
    small_NN_distance = small_c_dict["NN_distance"]
    if large_NN_distance < small_NN_distance:
        raise ValueError(
            f"large NN ({large_NN_distance}) < small NN "
            f"({small_NN_distance}) distance"
        )


def main() -> None:  # noqa: PLR0915
    """Run script."""
    ligand_dir = pathlib.Path("/home/atarzia/workingspace/cpl/ligand_analysis")
    calculation_dir = pathlib.Path(
        "/home/atarzia/workingspace/cpl/calculations"
    )
    figures_dir = pathlib.Path("/home/atarzia/workingspace/cpl/figures")
    ligand_dir.mkdir(exist_ok=True)
    calculation_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    ligand_db = atomlite.Database(ligand_dir / "ligands.db")
    pair_db = atomlite.Database(ligand_dir / "pairs.db")

    raise SystemExit("plot distributions of two flex measures from lC")
    raise SystemExit("plot distributions of two measures from shuhei paper")

    # Define minimum energies for all ligands.
    logging.info("remove this, I think")
    low_energy_values = {}
    for entry in ligand_db.get_entries():
        sres = entry.properties["conf_data"]
        min_energy = 1e24
        min_e_cid = 0
        for cid in sres:
            energy = sres[cid]["UFFEnergy;kj/mol"]
            if energy < min_energy:
                min_energy = energy
                min_e_cid = cid
        print(min_energy, entry.properties["min_energy;kj/mol"])
        low_energy_values[entry.key] = (min_e_cid, min_energy)
    print(low_energy_values)

    raise SystemExit("are they the same?")

    for ligand1, ligand2 in it.combinations(ligands, 2):
        logging.info(f"analysing {ligand1} and {ligand2}")
        key = f"{ligand1}_{ligand2}"
        print(key)
        if pair_db.has_property_entry(key):
            continue

        ligand1_entry = ligand_db.get_entry(ligand1)
        ligand1_confs = ligand1_entry["conf_data"]
        ligand2_entry = ligand_db.get_entry(ligand2)
        ligand2_confs = ligand2_entry["conf_data"]
        st = time.time()
        num_pairs = 0
        min_geom_score = 1e24
        print(len(ligand1_confs), len(ligand2_confs))
        pair_data = {}
        raise SystemExit
        # Iterate over the product of all conformers.
        for cid_1, cid_2 in it.product(ligand1_confs, ligand2_confs):
            cid_name = f"{cid_1}-{cid_2}"

            # Check strain.
            strain1 = (
                ligand1_confs[cid_1]["UFFEnergy;kj/mol"]
                - ligand1_entry["min_energy;kj/mol"]
            )
            strain2 = (
                ligand2_confs[cid_2]["UFFEnergy;kj/mol"]
                - ligand1_entry["min_energy;kj/mol"]
            )
            if (
                strain1 > EnvVariables.strain_cutoff
                or strain2 > EnvVariables.strain_cutoff
            ):
                continue

            print(strain1, strain2)
            # Calculate geom score for both sides together.
            c_dict1 = ligand1_confs[cid_1]
            c_dict2 = ligand2_confs[cid_2]
            print(c_dict1, c_dict2)

            logging.info(
                "should I pick large and small, or just define the alg so it does not matter?"
            )

            raise SystemExit

            # Calculate final geometrical properties.
            # T1.
            angle_dev = get_test_1(
                large_c_dict=large_c_dict,
                small_c_dict=small_c_dict,
            )
            # T2.
            length_dev = get_test_2(
                large_c_dict=large_c_dict,
                small_c_dict=small_c_dict,
                pdn_distance=vector_length(),
            )
            geom_score = abs(angle_dev - 1) + abs(length_dev - 1)

            min_geom_score = min((geom_score, min_geom_score))
            pair_data[cid_name] = {
                "geom_score": geom_score,
                "large_key": large_key,
                "small_key": small_key,
                "large_dihedral": large_c_dict["NCCN_dihedral"],
                "small_dihedral": small_c_dict["NCCN_dihedral"],
                "angle_deviation": angle_dev,
                "length_deviation": length_dev,
                "large_dict": large_c_dict,
                "small_dict": small_c_dict,
            }
            num_pairs += 1
        logging.info("need to handle the two different isomers of matching")
        print("t", pair_db.has_property_entry(key))
        entry = atomlite.PropertyEntry(
            key=key,
            properties={"pair_data": pair_data},
        )
        pair_db.update_entries(entries=entry)
        print("t2", pair_db.has_property_entry(key))
        ft = time.time()
        logging.info(
            f"time taken for pairing {ligand1}, {ligand2}: "
            f"{round(1000*(ft-st), 2)}ms "
            f"({round(1000*(ft-st)/num_pairs)}ms"
            f" per pair) - {num_pairs} pairs"
        )
        raise SystemExit

    # Figure in manuscript.
    plotting.gs_table(results_dict=pair_info, dihedral_cutoff=dihedral_cutoff)

    # Figure in manuscript.
    plotting.plot_all_ligand_pairings_simplified(
        results_dict=pair_info,
        dihedral_cutoff=dihedral_cutoff,
        outname=f"{figure_prefix}_all_lp_simpl.pdf",
    )
    plotting.plot_all_ligand_pairings(
        results_dict=pair_info,
        dihedral_cutoff=dihedral_cutoff,
        outname=f"{figure_prefix}_all_lp.png",
    )
    plotting.plot_all_ligand_pairings_2dhist(
        results_dict=pair_info,
        dihedral_cutoff=dihedral_cutoff,
        outname=f"{figure_prefix}_all_lp_2dhist.png",
    )
    plotting.plot_all_ligand_pairings_2dhist_fig5(
        results_dict=pair_info,
        dihedral_cutoff=dihedral_cutoff,
        outname=f"{figure_prefix}_all_lp_2dhist_fig5.pdf",
    )

    # Figures in SI.
    plotting.plot_all_geom_scores_simplified(
        results_dict=pair_info,
        outname=f"{figure_prefix}_all_pairs_simpl.png",
        dihedral_cutoff=dihedral_cutoff,
        experimental_ligand_outcomes=experimental_ligand_outcomes,
    )
    plotting.plot_all_ligand_pairings_conformers(
        results_dict=pair_info,
        structure_results=structure_results,
        dihedral_cutoff=dihedral_cutoff,
        outname=f"{figure_prefix}_all_lp_conformers.pdf",
    )

    # Figures not in SI.
    for pair_name in pair_info:
        small_l, large_l = pair_name.split(",")
        plotting.plot_ligand_pairing(
            results_dict=pair_info[pair_name],
            dihedral_cutoff=dihedral_cutoff,
            outname=f"{figure_prefix}_lp_{small_l}_{large_l}.png",
        )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    main()
