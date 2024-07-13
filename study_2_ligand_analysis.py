"""Script to build the ligand in this project."""

import logging
import pathlib
from rdkit.Chem import Draw
from collections import abc
import bbprep
import stk
import atomlite
import stko
import numpy as np
from rdkit.Chem import AllChem as rdkit
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


def build_ligand(
    building_blocks: abc.Sequence[stk.BuildingBlock],
    repeating_unit: str,
    ligand_name: str,
    ligand_dir: pathlib.Path,
    figures_dir: pathlib.Path,
    calculation_dir: pathlib.Path,
    ligand_db: atomlite.Database,
) -> stk.BuildingBlock:
    lowe_file = ligand_dir / f"{ligand_name}_lowe.mol"
    if lowe_file.exists():
        molecule = stk.BuildingBlock.init_from_file(lowe_file)
    else:
        # Build polymer.
        molecule = stk.ConstructedMolecule(
            topology_graph=stk.polymer.Linear(
                building_blocks=building_blocks,
                repeating_unit=repeating_unit,
                num_repeating_units=1,
                num_processes=1,
            )
        )
        molecule = stko.ETKDG().optimize(molecule)

    if not ligand_db.has_entry(key=ligand_name):
        explore_ligand(
            molecule=molecule,
            ligand_name=ligand_name,
            ligand_dir=ligand_dir,
            figures_dir=figures_dir,
            calculation_dir=calculation_dir,
            ligand_db=ligand_db,
        )
    return molecule


def generate_all_ligands(
    ligand_dir: pathlib.Path,
    figures_dir: pathlib.Path,
    calculation_dir: pathlib.Path,
    ligand_db: atomlite.Database,
) -> dict[str, stk.Molecule]:
    core_smiles = (
        # From 10.1002/anie.202106721
        "Brc1ccc(Br)cc1",
        "Brc1cccc(Br)c1",
        "Brc1ccc2[nH]c3ccc(Br)cc3c2c1",
        "Brc1ccc2ccc(Br)cc2c1",
        #####
        "C(#CBr)Br",
        "CC1=C(C=CC=C1Br)Br",
        "C1=CC=C2C(=C1)C(=C3C=CC=CC3=C2Br)Br",
        "CC1=C(C(=C(C(=C1Br)C)C)Br)C",
        "C1=C(SC(=C1)Br)Br",
        "C1=CC2=C(C=C1Br)C3=C(O2)C=CC(=C3)Br",
        "C1=C(OC(=C1)Br)Br",
        "C1=CC2=C(C=C1Br)C3=C(C2=O)C=CC(=C3)Br",
        "C1=CC2=C(C=C(C=C2)Br)C3=C1C=CC(=C3)Br",
        "CN1C2=C(C=C(C=C2)Br)C(=O)C3=C1C=CC(=C3)Br",
    )
    linker_smiles = (
        "C1=CC(=CC=C1Br)Br",
        "Brc1cccc(Br)c1",
        "BrC#CBr",
        "C1=CC=C2C(=C1)C(=C3C=CC=CC3=C2Br)Br",
    )
    binder_smiles = (
        # From 10.1002/anie.202106721
        "Brc1ccncc1",
        "Brc1cccnc1",
        # "BrC#Cc1cccnc1",
        # But removed the alkyne, cause its in the linker.
        "C1=CC2=C(C=CN=C2)C(=C1)Br",
        "C1=CC2=C(C=NC=C2)C(=C1)Br",
        "C1=CC2=C(C=CC=N2)C(=C1)Br",
        #####
        "CC1=C(C=NC=C1)Br",
        # "C1=CC(=CN=C1)C#CBr",
        # "C1C=C(C)C(C#CBr)=CN=1",
        "C1=CN(C=N1)Br",
        "CN1C=NC=C1Br",
        "C1=CC=C2C(=C1)C=NC=C2Br",
        "C1=CC=C2C(=C1)C=C(C=N2)Br",
        # "C1=CC=C2C(=C1)C(=C3C=CC=CC3=N2)Br",
    )

    all_ligands = {}
    for j, core in enumerate(core_smiles):
        for (i1, link1), (i2, link2) in it.combinations(enumerate(linker_smiles), r=2):
            for (k1, bind1), (k2, bind2) in it.combinations(
                enumerate(binder_smiles), r=2
            ):
                # Build all options from a-b-c-d-e configurations.
                options = {
                    "ace": {
                        "name": f"ace_{k1}-x-{j}-x-{k2}",
                        "bbs": (
                            stk.BuildingBlock(core, [stk.BromoFactory()]),
                            stk.BuildingBlock(bind1, [stk.BromoFactory()]),
                            stk.BuildingBlock(bind2, [stk.BromoFactory()]),
                        ),
                        "ru": "BAC",
                    },
                    "abe": {
                        "name": f"abe_{k1}-{i1}-x-x-{k2}",
                        "bbs": (
                            stk.BuildingBlock(link1, [stk.BromoFactory()]),
                            stk.BuildingBlock(bind1, [stk.BromoFactory()]),
                            stk.BuildingBlock(bind2, [stk.BromoFactory()]),
                        ),
                        "ru": "BAC",
                    },
                    "ade": {
                        "name": f"ade_{k1}-x-x-{i2}-{k2}",
                        "bbs": (
                            stk.BuildingBlock(link2, [stk.BromoFactory()]),
                            stk.BuildingBlock(bind1, [stk.BromoFactory()]),
                            stk.BuildingBlock(bind2, [stk.BromoFactory()]),
                        ),
                        "ru": "BAC",
                    },
                    "abce": {
                        "name": f"abce_{k1}-{i1}-{j}-x-{k2}",
                        "bbs": (
                            stk.BuildingBlock(link1, [stk.BromoFactory()]),
                            stk.BuildingBlock(bind1, [stk.BromoFactory()]),
                            stk.BuildingBlock(bind2, [stk.BromoFactory()]),
                            stk.BuildingBlock(core, [stk.BromoFactory()]),
                        ),
                        "ru": "BADC",
                    },
                    "adce": {
                        "name": f"adce_{k1}-x-{j}-{i2}-{k2}",
                        "bbs": (
                            stk.BuildingBlock(link2, [stk.BromoFactory()]),
                            stk.BuildingBlock(bind1, [stk.BromoFactory()]),
                            stk.BuildingBlock(bind2, [stk.BromoFactory()]),
                            stk.BuildingBlock(core, [stk.BromoFactory()]),
                        ),
                        "ru": "BADC",
                    },
                    "acbe": {
                        "name": f"acbe_{k1}-{i1}-{j}-x-{k2}",
                        "bbs": (
                            stk.BuildingBlock(link1, [stk.BromoFactory()]),
                            stk.BuildingBlock(bind1, [stk.BromoFactory()]),
                            stk.BuildingBlock(bind2, [stk.BromoFactory()]),
                            stk.BuildingBlock(core, [stk.BromoFactory()]),
                        ),
                        "ru": "BDAC",
                    },
                    "acde": {
                        "name": f"acde_{k1}-x-{j}-{i2}-{k2}",
                        "bbs": (
                            stk.BuildingBlock(link2, [stk.BromoFactory()]),
                            stk.BuildingBlock(bind1, [stk.BromoFactory()]),
                            stk.BuildingBlock(bind2, [stk.BromoFactory()]),
                            stk.BuildingBlock(core, [stk.BromoFactory()]),
                        ),
                        "ru": "BDAC",
                    },
                    "abcde": {
                        "name": f"abcde_{k1}-{i1}-{j}-{i2}-{k2}",
                        "bbs": (
                            stk.BuildingBlock(link1, [stk.BromoFactory()]),
                            stk.BuildingBlock(bind1, [stk.BromoFactory()]),
                            stk.BuildingBlock(bind2, [stk.BromoFactory()]),
                            stk.BuildingBlock(core, [stk.BromoFactory()]),
                            stk.BuildingBlock(link2, [stk.BromoFactory()]),
                        ),
                        "ru": "BADEC",
                    },
                    "adcbe": {
                        "name": f"adcbe_{k1}-{i1}-{j}-{i2}-{k2}",
                        "bbs": (
                            stk.BuildingBlock(link2, [stk.BromoFactory()]),
                            stk.BuildingBlock(bind1, [stk.BromoFactory()]),
                            stk.BuildingBlock(bind2, [stk.BromoFactory()]),
                            stk.BuildingBlock(core, [stk.BromoFactory()]),
                            stk.BuildingBlock(link1, [stk.BromoFactory()]),
                        ),
                        "ru": "BADEC",
                    },
                    "abcbe": {
                        "name": f"abcbe_{k1}-{i1}-{j}-x-{k2}",
                        "bbs": (
                            stk.BuildingBlock(link1, [stk.BromoFactory()]),
                            stk.BuildingBlock(bind1, [stk.BromoFactory()]),
                            stk.BuildingBlock(bind2, [stk.BromoFactory()]),
                            stk.BuildingBlock(core, [stk.BromoFactory()]),
                        ),
                        "ru": "BADAC",
                    },
                    "adcde": {
                        "name": f"adcde_{k1}-x-{j}-{i2}-{k2}",
                        "bbs": (
                            stk.BuildingBlock(link2, [stk.BromoFactory()]),
                            stk.BuildingBlock(bind1, [stk.BromoFactory()]),
                            stk.BuildingBlock(bind2, [stk.BromoFactory()]),
                            stk.BuildingBlock(core, [stk.BromoFactory()]),
                        ),
                        "ru": "BADAC",
                    },
                }

                for option in options:
                    ligand_name = options[option]["name"]
                    if len([i for i in ligand_name if i == "-"]) != 4:
                        print(ligand_name, option)
                        raise RuntimeError
                    all_ligands[ligand_name] = build_ligand(
                        ligand_name=ligand_name,
                        building_blocks=options[option]["bbs"],
                        repeating_unit=options[option]["ru"],
                        ligand_dir=ligand_dir,
                        figures_dir=figures_dir,
                        calculation_dir=calculation_dir,
                        ligand_db=ligand_db,
                    )

    return all_ligands


def explore_ligand(
    molecule: stk.Molecule,
    ligand_name: str,
    ligand_dir: pathlib.Path,
    figures_dir: pathlib.Path,
    calculation_dir: pathlib.Path,
    ligand_db: atomlite.Database,
):
    rdkit_mol = rdkit.MolFromSmiles(stk.Smiles().get_key(molecule))
    Draw.MolToFile(rdkit_mol, figures_dir / f"{ligand_name}_2d.png", size=(300, 300))

    st = time.time()
    conf_dir = ligand_dir / f"confs_{ligand_name}"
    conf_dir.mkdir(exist_ok=True)

    logging.info(f"building conformer ensemble of {ligand_name}")

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
        conf_opt_file_name = f"{ligand_name}_c{cid}_cuff.mol"

        # Update stk_mol to conformer geometry.
        new_mol = update_from_rdkit_conf(stk_mol=molecule, rdk_mol=confs, conf_id=cid)
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
            new_mol.write(ligand_dir / f"{ligand_name}_lowe.mol")

        analyser = stko.molecule_analysis.DitopicThreeSiteAnalyser()

        lig_conf_data[cid] = {
            "NcentroidN_angle": analyser.get_binder_centroid_angle(new_mol),
            "NCCN_dihedral": analyser.get_binder_adjacent_torsion(new_mol),
            "NN_distance": analyser.get_binder_distance(new_mol),
            "NN_BCN_angles": analyser.get_binder_angles(new_mol),
            "UFFEnergy;kj/mol": energy * 4.184,
        }
        num_confs += 1

    print(
        ligand_name,
        {
            "conf_data": lig_conf_data,
            "min_energy;kj/mol": min_energy,
            "ligand_pattern": ligand_name.split("_")[0],
            "composition_pattern": ligand_name.split("_")[1],
        },
    )
    entry = atomlite.Entry.from_rdkit(
        key=ligand_name,
        molecule=stk.BuildingBlock.init_from_file(
            ligand_dir / f"{ligand_name}_lowe.mol"
        ).to_rdkit_mol(),
        properties={
            "conf_data": lig_conf_data,
            "min_energy;kj/mol": min_energy,
            "ligand_pattern": ligand_name.split("_")[0],
            "composition_pattern": ligand_name.split("_")[1],
        },
    )
    ligand_db.add_entries(entry)

    logging.info(
        f"{num_confs} confs generated for {ligand_name} in {round(time.time()-st, 2)}s"
    )
    raise SystemExit


def main():
    ligand_dir = pathlib.Path("/home/atarzia/workingspace/cpl/ligand_analysis")
    calculation_dir = pathlib.Path("/home/atarzia/workingspace/cpl/calculations")
    figures_dir = pathlib.Path("/home/atarzia/workingspace/cpl/figures")
    ligand_dir.mkdir(exist_ok=True)
    calculation_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    ligand_db = atomlite.Database(ligand_dir / "ligands.db")
    pair_db = atomlite.Database(ligand_dir / "pairs.db")

    dihedral_cutoff = 10
    strain_cutoff = 5

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
            if strain1 > strain_cutoff or strain2 > strain_cutoff:
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
