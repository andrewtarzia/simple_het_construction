"""Script to build the ligand in this project."""

import itertools as it
import logging
import pathlib
import time
from collections import abc

import atomlite
import bbprep
import stk
import stko
from rdkit.Chem import AllChem as rdkit
from rdkit.Chem import Draw
from utilities import update_from_rdkit_conf


def build_ligand(
    building_blocks: abc.Sequence[stk.BuildingBlock],
    repeating_unit: str,
    ligand_name: str,
    ligand_dir: pathlib.Path,
    figures_dir: pathlib.Path,
    calculation_dir: pathlib.Path,
    ligand_db: atomlite.Database,
) -> None:
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


def generate_all_ligands(
    ligand_dir: pathlib.Path,
    figures_dir: pathlib.Path,
    calculation_dir: pathlib.Path,
    ligand_db: atomlite.Database,
) -> None:
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

    count = 0
    total_expected = 41580
    for j, core in enumerate(core_smiles):
        logging.info(
            "c%s: at count %s of %s (%s)",
            j,
            count,
            total_expected,
            round(count / total_expected, 2),
        )
        for (i1, link1), (i2, link2) in it.combinations(
            enumerate(linker_smiles), r=2
        ):
            logging.info(
                "l%s, l%s: at %s of %s (%s)",
                i1,
                i2,
                count,
                total_expected,
                round(count / total_expected, 2),
            )
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

                    build_ligand(
                        ligand_name=ligand_name,
                        building_blocks=options[option]["bbs"],
                        repeating_unit=options[option]["ru"],
                        ligand_dir=ligand_dir,
                        figures_dir=figures_dir,
                        calculation_dir=calculation_dir,
                        ligand_db=ligand_db,
                    )
                    count += 1


def explore_ligand(
    molecule: stk.Molecule,
    ligand_name: str,
    ligand_dir: pathlib.Path,
    figures_dir: pathlib.Path,
    calculation_dir: pathlib.Path,
    ligand_db: atomlite.Database,
) -> None:
    rdkit_mol = rdkit.MolFromSmiles(stk.Smiles().get_key(molecule))
    Draw.MolToFile(
        rdkit_mol, figures_dir / f"{ligand_name}_2d.png", size=(300, 300)
    )

    st = time.time()
    conf_dir = ligand_dir / f"confs_{ligand_name}"
    conf_dir.mkdir(exist_ok=True)

    logging.info(f"building conformer ensemble of {ligand_name}")

    confs = molecule.to_rdkit_mol()
    etkdg = rdkit.srETKDGv3()
    etkdg.randomSeed = 1000
    etkdg.pruneRmsThresh = 0.2
    cids = rdkit.EmbedMultipleConfs(mol=confs, numConfs=500, params=etkdg)

    lig_conf_data = {}
    num_confs = 0
    min_energy = 1e24
    for cid in cids:
        conf_opt_file_name = f"{ligand_name}_c{cid}_cuff.mol"

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


def main() -> None:
    """Run script."""
    ligand_dir = pathlib.Path("/home/atarzia/workingspace/cpl/ligand_analysis")
    calculation_dir = pathlib.Path(
        "/home/atarzia/workingspace/cpl/calculations"
    )
    figures_dir = pathlib.Path(
        "/home/atarzia/workingspace/cpl/figures/ligands_2d"
    )
    ligand_dir.mkdir(exist_ok=True)
    calculation_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True, parents=True)

    ligand_db = atomlite.Database(ligand_dir / "ligands.db")

    # Generate all ligands from core, binder and connector pools.
    generate_all_ligands(
        ligand_dir=ligand_dir,
        figures_dir=figures_dir,
        calculation_dir=calculation_dir,
        ligand_db=ligand_db,
    )
    logging.info("built %s ligands", ligand_db.num_entries)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    main()
