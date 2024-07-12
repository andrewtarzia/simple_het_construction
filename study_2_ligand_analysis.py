"""Script to build the ligand in this project."""

import logging
import pathlib
from rdkit.Chem import Draw
import bbprep
import os
import json
import stk
import stko
import numpy as np
from rdkit.Chem import AllChem as rdkit
import itertools
import time

from utilities import update_from_rdkit_conf
import itertools as it


def vector_length():
    """
    Mean value of bond distance to use in candidate selection.

    """
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


def conformer_generation_uff(
    molecule,
    name,
    conf_data_file,
    calc_dir,
    ligand_dir,
):
    """Build a large conformer ensemble with UFF optimisation."""
    conf_dir = ligand_dir / f"confs_{name}"
    conf_dir.mkdir(exist_ok=True)

    logging.info(f"building conformer ensemble of {name}")

    confs = molecule.to_rdkit_mol()
    etkdg = rdkit.srETKDGv3()
    etkdg.randomSeed = 1000
    etkdg.pruneRmsThresh = 0.2
    cids = rdkit.EmbedMultipleConfs(
        mol=confs,
        numConfs=500,
        params=etkdg,
    )

    lig_conf_data = {}
    num_confs = 0
    min_energy = 1e24
    for cid in cids:
        conf_opt_file_name = f"{name}_c{cid}_cuff.mol"

        # Update stk_mol to conformer geometry.
        new_mol = update_from_rdkit_conf(
            stk_mol=molecule, rdk_mol=confs, conf_id=cid
        )
        # Need to define the functional groups.
        new_mol = stk.BuildingBlock.init_from_molecule(
            molecule=new_mol,
            functional_groups=stko.functional_groups.ThreeSiteFactory(
                smarts="[#6]~[#7X2]~[#6]", bonders=(1,), deleters=()
            ),
        )
        # Only get two FGs.
        new_mol = bbprep.FurthestFGs().modify(
            building_block=new_mol,
            desired_functional_groups=2,
        )

        new_mol = stko.UFF().optimize(mol=new_mol)
        energy = stko.UFFEnergy().get_energy(new_mol)
        new_mol.write(conf_dir / conf_opt_file_name)
        if energy < min_energy:
            min_energy = energy
            new_mol.write(ligand_dir / f"{name}_lowe.mol")

        analyser = stko.molecule_analysis.DitopicThreeSiteAnalyser()

        lig_conf_data[cid] = {
            "NcentroidN_angle": analyser.get_binder_centroid_angle(new_mol),
            "NCCN_dihedral": analyser.get_binder_adjacent_torsion(new_mol),
            "NN_distance": analyser.get_binder_distance(new_mol),
            "NN_BCN_angles": analyser.get_binder_angles(new_mol),
            "UFFEnergy;kj/mol": energy * 4.184,
        }
        num_confs += 1

    logging.info(f"{num_confs} conformers generated for {name}")

    with open(conf_data_file, "w") as f:
        json.dump(lig_conf_data, f)


def generate_converging_ligands() -> dict[str, stk.Molecule]:

    return {
        "d1": stk.BuildingBlock(
            smiles="C1=CC(=CC(=C1)C#CC2=CN=CC=C2)C#CC3=CN=CC=C3"
        )
    }


def generate_diverging_ligands(
    ligand_dir: pathlib.Path,
    figures_dir: pathlib.Path,
    calculation_dir: pathlib.Path,
) -> dict[str, stk.Molecule]:
    core_smiles = (
        # From 10.1002/anie.202106721
        "Brc1ccc(Br)cc1",
        "Brc1cccc(Br)c1",
        "Brc1ccc2[nH]c3ccc(Br)cc3c2c1",
        "Brc1ccc2ccc(Br)cc2c1",
    )
    linker_smiles = ("C1=CC(=CC=C1Br)Br",)
    binder_smiles = (
        # From 10.1002/anie.202106721
        "Brc1ccncc1",
        "Brc1cccnc1",
        "BrC#Cc1cccnc1",
        "BrC#Cc1cccc2cnccc12",
        "BrC#Cc1cccc2ccncc12",
        "BrC#Cc1cccc2ncccc12",
    )

    all_ligands = {}
    for (j, core), (i, link) in it.product(
        enumerate(core_smiles), enumerate(linker_smiles)
    ):
        for (k1, bind1), (k2, bind2) in it.combinations(
            enumerate(binder_smiles), r=2
        ):

            # Build ABA.
            ligand_name = f"aba_{j}{k1}{k2}"
            lowe_file = ligand_dir / f"{ligand_name}_lowe.mol"
            if lowe_file.exists():
                molecule = stk.BuildingBlock.init_from_file(lowe_file)
            else:
                # Build polymer.
                molecule = stk.ConstructedMolecule(
                    topology_graph=stk.polymer.Linear(
                        building_blocks=(
                            stk.BuildingBlock(core, [stk.BromoFactory()]),
                            stk.BuildingBlock(bind1, [stk.BromoFactory()]),
                            stk.BuildingBlock(bind2, [stk.BromoFactory()]),
                        ),
                        repeating_unit="BAC",
                        num_repeating_units=1,
                        orientations=(0, 0, 0),
                        num_processes=1,
                    )
                )
                # Optimise with ETKDG.
                molecule = stko.ETKDG().optimize(molecule)
                explore_ligand(
                    molecule=molecule,
                    ligand_name=ligand_name,
                    ligand_dir=ligand_dir,
                    figures_dir=figures_dir,
                    calculation_dir=calculation_dir,
                )
            all_ligands[ligand_name] = molecule

            # Build ABCBA
            ligand_name = f"abcba_{j}{i}{k1}{k2}"
            lowe_file = ligand_dir / f"{ligand_name}_lowe.mol"
            if lowe_file.exists():
                molecule = stk.BuildingBlock.init_from_file(lowe_file)
            else:
                # Build polymer.
                molecule = stk.ConstructedMolecule(
                    topology_graph=stk.polymer.Linear(
                        building_blocks=(
                            stk.BuildingBlock(core, [stk.BromoFactory()]),
                            stk.BuildingBlock(link, [stk.BromoFactory()]),
                            stk.BuildingBlock(bind1, [stk.BromoFactory()]),
                            stk.BuildingBlock(bind2, [stk.BromoFactory()]),
                        ),
                        repeating_unit="CBABD",
                        num_repeating_units=1,
                        orientations=(0, 0, 0, 0, 0),
                        num_processes=1,
                    )
                )
                # Optimise with ETKDG.
                molecule = stko.ETKDG().optimize(molecule)
                explore_ligand(
                    molecule=molecule,
                    ligand_name=ligand_name,
                    ligand_dir=ligand_dir,
                    figures_dir=figures_dir,
                    calculation_dir=calculation_dir,
                )
            all_ligands[ligand_name] = molecule

    return all_ligands


def explore_ligand(
    molecule: stk.Molecule,
    ligand_name: str,
    ligand_dir: pathlib.Path,
    figures_dir: pathlib.Path,
    calculation_dir: pathlib.Path,
):

    confuff_data_file = ligand_dir / f"{ligand_name}_conf_uff_data.json"
    rdkit_mol = rdkit.MolFromSmiles(stk.Smiles().get_key(molecule))
    Draw.MolToFile(
        rdkit_mol, figures_dir / f"{ligand_name}_2d.png", size=(300, 300)
    )

    if not confuff_data_file.exists():
        st = time.time()
        conformer_generation_uff(
            molecule=molecule,
            name=ligand_name,
            ligand_dir=ligand_dir,
            conf_data_file=confuff_data_file,
            calc_dir=calculation_dir,
        )
        logging.info(
            f"time taken for conf gen of {ligand_name}: "
            f"{round(time.time()-st, 2)}s"
        )


def main():

    ligand_dir = pathlib.Path("/home/atarzia/workingspace/cpl/ligand_analysis")
    calculation_dir = pathlib.Path(
        "/home/atarzia/workingspace/cpl/calculations"
    )
    figures_dir = pathlib.Path("/home/atarzia/workingspace/cpl/figures")
    ligand_dir.mkdir(exist_ok=True)
    calculation_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    dihedral_cutoff = 10
    strain_cutoff = 5

    # Generate all ligands from core, binder and connector pools.
    diverging = generate_diverging_ligands(
        ligand_dir=ligand_dir,
        figures_dir=figures_dir,
        calculation_dir=calculation_dir,
    )
    converging = generate_converging_ligands()
    logging.info(
        "build %s diverging and %s converging", len(diverging), len(converging)
    )

    raise SystemExit

    res_file = os.path.join(_wd, "all_ligand_res.json")
    pair_file = os.path.join(_wd, "all_pair_res.json")
    conf_data_suffix = "conf_uff_data"
    figure_prefix = "etkdg"

    structure_results = {}
    if os.path.exists(res_file):
        with open(res_file, "r") as f:
            structure_results = json.load(f)
    else:
        for ligand in ligand_smiles():
            st = time.time()
            structure_results[ligand] = {}
            conf_data_file = _wd / f"{ligand}_{conf_data_suffix}.json"
            with open(conf_data_file, "r") as f:
                property_dict = json.load(f)

            for cid in property_dict:
                pdi = property_dict[cid]["NN_BCN_angles"]
                # 180 - angle, to make it the angle toward the binding
                # interaction. Minus 90  to convert to the bite-angle.
                ba = ((180 - pdi["NN_BCN1"]) - 90) + (
                    (180 - pdi["NN_BCN2"]) - 90
                )
                property_dict[cid]["bite_angle"] = ba

            structure_results[ligand] = property_dict

            for yprop in yproperties:
                plotting.plot_single_distribution(
                    results_dict=structure_results[ligand],
                    outname=f"{figure_prefix}_d_{ligand}_{yprop}",
                    yproperty=yprop,
                )

            logging.info(
                f"time taken for getting struct results {ligand}: "
                f"{round(time.time()-st, 2)}s"
            )
        with open(res_file, "w") as f:
            json.dump(structure_results, f, indent=4)

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

    if os.path.exists(pair_file):
        logging.info(f"loading {pair_file}")
        with open(pair_file, "r") as f:
            pair_info = json.load(f)
    else:
        pair_info = {}
        min_geom_scores = {}
        for small_l, large_l in ligand_pairings:
            logging.info(f"analysing {small_l} and {large_l}")
            st = time.time()
            min_geom_score = 1e24
            pair_name = ",".join((small_l, large_l))
            pair_info[pair_name] = {}
            small_l_dict = structure_results[small_l]
            large_l_dict = structure_results[large_l]

            # Iterate over the product of all conformers.
            for small_cid, large_cid in itertools.product(
                small_l_dict, large_l_dict
            ):
                cid_name = ",".join((small_cid, large_cid))
                # Calculate geom score for both sides together.
                large_c_dict = large_l_dict[large_cid]
                small_c_dict = small_l_dict[small_cid]

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

                small_energy = small_l_dict[small_cid]["UFFEnergy;kj/mol"]
                small_strain = small_energy - low_energy_values[small_l][1]
                large_energy = large_l_dict[large_cid]["UFFEnergy;kj/mol"]
                large_strain = large_energy - low_energy_values[large_l][1]
                if (
                    small_strain > strain_cutoff
                    or large_strain > strain_cutoff
                ):
                    continue
                # total_strain = large_strain + small_strain

                min_geom_score = min((geom_score, min_geom_score))
                pair_info[pair_name][cid_name] = {
                    "geom_score": geom_score,
                    # "swapped_LS": swapped_LS,
                    # "converging": converging,
                    # "diverging": diverging,
                    "large_dihedral": large_c_dict["NCCN_dihedral"],
                    "small_dihedral": small_c_dict["NCCN_dihedral"],
                    "angle_deviation": angle_dev,
                    "length_deviation": length_dev,
                    # "small_NCN_angle": small_c_dict["NcentroidN_angle"],
                    # "large_NCN_angle": large_c_dict["NcentroidN_angle"],
                    # "small_energy": small_energy,
                    # "large_energy": large_energy,
                    # "small_strain": small_strain,
                    # "large_strain": large_strain,
                    # "total_strain": total_strain,
                }
            min_geom_scores[pair_name] = round(min_geom_score, 2)
            ft = time.time()
            logging.info(
                f"time taken for pairing {small_l}, {large_l}: "
                f"{round(1000*(ft-st), 2)}ms "
                f"({round(1000*(ft-st)/len(pair_info[pair_name]), 2)}ms"
                f" per pair) - {len(pair_info[pair_name])} pairs"
            )

        logging.info(f"Min. geom scores for each pair:\n {min_geom_scores}")

        with open(pair_file, "w") as f:
            json.dump(pair_info, f, indent=4)

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
