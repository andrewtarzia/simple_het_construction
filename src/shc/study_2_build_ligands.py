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
from rdkit.Chem import AllChem as rdkit  # noqa: N813
from rdkit.Chem import Draw
from utilities import update_from_rdkit_conf


def symmetry_check(
    building_blocks: abc.Sequence[stk.BuildingBlock],
    composition: str,
    repeating_unit: str,
) -> bool:
    """Check if a ligand will be symmetric."""
    base = ord("A")
    ru = tuple(ord(letter) - base for letter in repeating_unit)
    bb_smiles = [stk.Smiles().get_key(i) for i in building_blocks]

    if composition == "ae":
        return bb_smiles[0] == bb_smiles[1]

    if composition in ("ace", "abe", "ace", "ade"):
        return bb_smiles[ru[0]] == bb_smiles[ru[2]]

    if composition in ("abce", "adce", "acbe", "acde"):
        outers = bb_smiles[ru[0]] == bb_smiles[ru[3]]
        inners = bb_smiles[ru[1]] == bb_smiles[ru[2]]
        return outers and inners

    if composition in ("abca", "ebce", "adca", "edce"):
        return bb_smiles[ru[1]] == bb_smiles[ru[2]]

    if composition in ("abcde", "adcbe", "abcbe", "adcde"):
        outers = bb_smiles[ru[0]] == bb_smiles[ru[4]]
        inners = bb_smiles[ru[1]] == bb_smiles[ru[3]]
        return outers and inners

    if composition in ("abcda", "ebcde"):
        return bb_smiles[ru[1]] == bb_smiles[ru[3]]

    msg = f"missing definition for {composition}"
    raise NotImplementedError(msg)


def normalise_names(name: str) -> str:
    """Normalise names.

    No normalisation actually needed based on iteration.
    """
    # Flip order of building blocks if binder 1 is > id than binder 2.
    bbs = name.split("_")[1].split("-")

    if bbs[0] == "x" or bbs[-1] == "x":
        return name

    return name


def passes_dedupe(molecule: stk.Molecule, ligand_dir: pathlib.Path) -> bool:
    """Check if ligand is in dedupe databases."""
    key_db = atomlite.Database(ligand_dir / "keys.db")
    smiles = stk.Smiles().get_key(molecule)
    if key_db.has_entry(smiles):
        return False
    inchi = stk.Inchi().get_key(molecule)
    if key_db.has_entry(inchi):
        molecule.write("1i.mol")
        stk.BuildingBlock.init_from_rdkit_mol(
            atomlite.json_to_rdkit(key_db.get_entry(inchi).molecule)
        ).write("2i.mol")
        raise SystemExit
        return False
    inchi_key = stk.InchiKey().get_key(molecule)
    if key_db.has_entry(inchi_key):
        molecule.write("1ik.mol")
        stk.BuildingBlock.init_from_rdkit_mol(
            atomlite.json_to_rdkit(key_db.get_entry(inchi_key).molecule)
        ).write("2ik.mol")
        raise SystemExit
        return False

    return True


def update_keys_db(molecule: stk.Molecule, ligand_dir: pathlib.Path) -> None:
    """Update keys db."""
    key_db = atomlite.Database(ligand_dir / "keys.db")
    smiles = stk.Smiles().get_key(molecule)
    inchi = stk.Inchi().get_key(molecule)
    inchi_key = stk.InchiKey().get_key(molecule)
    key_db.add_entries(
        entries=(
            atomlite.Entry(
                key=smiles,
                molecule=atomlite.json_from_rdkit(molecule.to_rdkit_mol()),
            ),
            atomlite.Entry(
                key=inchi,
                molecule=atomlite.json_from_rdkit(molecule.to_rdkit_mol()),
            ),
            atomlite.Entry(
                key=inchi_key,
                molecule=atomlite.json_from_rdkit(molecule.to_rdkit_mol()),
            ),
        )
    )


def build_ligand(  # noqa: PLR0913
    building_blocks: abc.Sequence[stk.BuildingBlock],
    repeating_unit: str,
    ligand_name: str,
    ligand_dir: pathlib.Path,
    figures_dir: pathlib.Path,
    ligand_db: atomlite.Database,
    deduped_db: atomlite.Database,
) -> None:
    """Build a ligand."""
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

    if passes_dedupe(molecule=molecule, ligand_dir=ligand_dir):
        Draw.MolToFile(
            rdkit.MolFromSmiles(stk.Smiles().get_key(molecule)),
            figures_dir / f"{ligand_name}_2d_new.png",
            size=(300, 300),
        )
        molecule = stko.ETKDG().optimize(molecule)
        if ligand_db.has_entry(key=ligand_name) and not deduped_db.has_entry(
            key=ligand_name
        ):
            # Bring to deduped db.
            lig_entry = ligand_db.get_entry(key=ligand_name)
            deduped_db.add_entries(lig_entry)

        # Check if in db.
        if not deduped_db.has_entry(key=ligand_name):
            explore_ligand(
                molecule=molecule,
                ligand_name=ligand_name,
                ligand_dir=ligand_dir,
                ligand_db=deduped_db,
            )

        update_keys_db(molecule, ligand_dir)


def generate_all_ligands(
    ligand_dir: pathlib.Path,
    figures_dir: pathlib.Path,
    components_dir: pathlib.Path,
    ligand_db: atomlite.Database,
    deduped_db: atomlite.Database,
) -> None:
    """Iterate and generate through all components."""
    core_smiles = {
        # From 10.1002/anie.202106721
        0: "Brc1ccc(Br)cc1",
        1: "Brc1cccc(Br)c1",
        2: "Brc1ccc2[nH]c3ccc(Br)cc3c2c1",
        3: "Brc1ccc2ccc(Br)cc2c1",
        #####
        4: "C(#CBr)Br",
        # 5: "CC1=C(C=CC=C1Br)Br",
        # 6: "C1=CC=C2C(=C1)C(=C3C=CC=CC3=C2Br)Br",
        # 7: "CC1=C(C(=C(C(=C1Br)C)C)Br)C",
        8: "C1=C(SC(=C1)Br)Br",
        9: "C1=CC2=C(C=C1Br)C3=C(O2)C=CC(=C3)Br",
        10: "C1=C(OC(=C1)Br)Br",
        11: "C1=CC2=C(C=C1Br)C3=C(C2=O)C=CC(=C3)Br",
        12: "C1=CC2=C(C=C(C=C2)Br)C3=C1C=CC(=C3)Br",
        13: "CN1C2=C(C=C(C=C2)Br)C(=O)C3=C1C=CC(=C3)Br",
        14: "C1=CC=C(C(=C1)Br)Br",
    }
    linker_smiles = {
        0: "C1=CC(=CC=C1Br)Br",
        1: "Brc1cccc(Br)c1",
        2: "BrC#CBr",
        # 3: "C1=CC=C2C(=C1)C(=C3C=CC=CC3=C2Br)Br",
        4: "C1=CC=C(C(=C1)Br)Br",
    }
    binder_smiles = {
        # From 10.1002/anie.202106721
        0: "Brc1ccncc1",
        1: "Brc1cccnc1",
        # "BrC#Cc1cccnc1",
        # But removed the alkyne, cause its in the linker.
        2: "C1=CC2=C(C=CN=C2)C(=C1)Br",
        3: "C1=CC2=C(C=NC=C2)C(=C1)Br",
        4: "C1=CC2=C(C=CC=N2)C(=C1)Br",
        #####
        # 5: "CC1=C(C=NC=C1)Br",
        # "C1=CC(=CN=C1)C#CBr",  # noqa: ERA001
        # "C1C=C(C)C(C#CBr)=CN=1",  # noqa: ERA001
        6: "C1=CN(C=N1)Br",
        7: "CN1C=NC=C1Br",
        8: "C1=CC=C2C(=C1)C=NC=C2Br",
        9: "C1=CC=C2C(=C1)C=C(C=N2)Br",
        # "C1=CC=C2C(=C1)C(=C3C=CC=CC3=N2)Br",  # noqa: ERA001
    }

    for lig in core_smiles:
        rdkit_mol = rdkit.MolFromSmiles(core_smiles[lig])
        Draw.MolToFile(
            rdkit_mol, components_dir / f"core_{lig}_2d.png", size=(300, 300)
        )

    for lig in linker_smiles:
        rdkit_mol = rdkit.MolFromSmiles(linker_smiles[lig])
        Draw.MolToFile(
            rdkit_mol, components_dir / f"link_{lig}_2d.png", size=(300, 300)
        )

    for lig in binder_smiles:
        rdkit_mol = rdkit.MolFromSmiles(binder_smiles[lig])
        Draw.MolToFile(
            rdkit_mol, components_dir / f"bind_{lig}_2d.png", size=(300, 300)
        )

    count = 0

    for core_name in core_smiles:
        core = core_smiles[core_name]
        logging.info("c%s: at count %s", core_name, count)

        for link1_name, link2_name in it.combinations(linker_smiles, r=2):
            link1 = linker_smiles[link1_name]
            link2 = linker_smiles[link2_name]
            logging.info("l%s, l%s: at %s", link1_name, link2_name, count)

            for bind1_name, bind2_name in it.combinations(binder_smiles, r=2):
                bind1 = binder_smiles[bind1_name]
                bind2 = binder_smiles[bind2_name]
                # Build all options from a-b-c-d-e configurations.
                options = {
                    "ae": {
                        "name": (f"ae_{bind1_name}-x-x-x-{bind2_name}"),
                        "bbs": (
                            stk.BuildingBlock(bind1, [stk.BromoFactory()]),
                            stk.BuildingBlock(bind2, [stk.BromoFactory()]),
                        ),
                        "ru": "BA",
                    },
                    "ace": {
                        "name": (
                            f"ace_{bind1_name}-x-{core_name}-x-{bind2_name}"
                        ),
                        "bbs": (
                            stk.BuildingBlock(core, [stk.BromoFactory()]),
                            stk.BuildingBlock(bind1, [stk.BromoFactory()]),
                            stk.BuildingBlock(bind2, [stk.BromoFactory()]),
                        ),
                        "ru": "BAC",
                    },
                    "abe": {
                        "name": (
                            f"abe_{bind1_name}-{link1_name}-x-x-{bind2_name}"
                        ),
                        "bbs": (
                            stk.BuildingBlock(link1, [stk.BromoFactory()]),
                            stk.BuildingBlock(bind1, [stk.BromoFactory()]),
                            stk.BuildingBlock(bind2, [stk.BromoFactory()]),
                        ),
                        "ru": "BAC",
                    },
                    "ade": {
                        "name": (
                            f"ade_{bind1_name}-x-x-{link2_name}-{bind2_name}"
                        ),
                        "bbs": (
                            stk.BuildingBlock(link2, [stk.BromoFactory()]),
                            stk.BuildingBlock(bind1, [stk.BromoFactory()]),
                            stk.BuildingBlock(bind2, [stk.BromoFactory()]),
                        ),
                        "ru": "BAC",
                    },
                    "abce": {
                        "name": (
                            f"abce_{bind1_name}-{link1_name}-{core_name}-x-"
                            f"{bind2_name}"
                        ),
                        "bbs": (
                            stk.BuildingBlock(link1, [stk.BromoFactory()]),
                            stk.BuildingBlock(bind1, [stk.BromoFactory()]),
                            stk.BuildingBlock(bind2, [stk.BromoFactory()]),
                            stk.BuildingBlock(core, [stk.BromoFactory()]),
                        ),
                        "ru": "BADC",
                    },
                    "adce": {
                        "name": (
                            f"adce_{bind1_name}-x-{core_name}-{link2_name}-"
                            f"{bind2_name}"
                        ),
                        "bbs": (
                            stk.BuildingBlock(link2, [stk.BromoFactory()]),
                            stk.BuildingBlock(bind1, [stk.BromoFactory()]),
                            stk.BuildingBlock(bind2, [stk.BromoFactory()]),
                            stk.BuildingBlock(core, [stk.BromoFactory()]),
                        ),
                        "ru": "BADC",
                    },
                    "acbe": {
                        "name": (
                            f"acbe_{bind1_name}-{link1_name}-{core_name}-x-"
                            f"{bind2_name}"
                        ),
                        "bbs": (
                            stk.BuildingBlock(link1, [stk.BromoFactory()]),
                            stk.BuildingBlock(bind1, [stk.BromoFactory()]),
                            stk.BuildingBlock(bind2, [stk.BromoFactory()]),
                            stk.BuildingBlock(core, [stk.BromoFactory()]),
                        ),
                        "ru": "BDAC",
                    },
                    "acde": {
                        "name": (
                            f"acde_{bind1_name}-x-{core_name}-{link2_name}-"
                            f"{bind2_name}"
                        ),
                        "bbs": (
                            stk.BuildingBlock(link2, [stk.BromoFactory()]),
                            stk.BuildingBlock(bind1, [stk.BromoFactory()]),
                            stk.BuildingBlock(bind2, [stk.BromoFactory()]),
                            stk.BuildingBlock(core, [stk.BromoFactory()]),
                        ),
                        "ru": "BDAC",
                    },
                    "abcde": {
                        "name": (
                            f"abcde_{bind1_name}-{link1_name}-{core_name}-"
                            f"{link2_name}-{bind2_name}"
                        ),
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
                        "name": (
                            f"adcbe_{bind1_name}-{link1_name}-{core_name}-"
                            f"{link2_name}-{bind2_name}"
                        ),
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
                        "name": (
                            f"abcbe_{bind1_name}-{link1_name}-{core_name}-x-"
                            f"{bind2_name}"
                        ),
                        "bbs": (
                            stk.BuildingBlock(link1, [stk.BromoFactory()]),
                            stk.BuildingBlock(bind1, [stk.BromoFactory()]),
                            stk.BuildingBlock(bind2, [stk.BromoFactory()]),
                            stk.BuildingBlock(core, [stk.BromoFactory()]),
                        ),
                        "ru": "BADAC",
                    },
                    "adcde": {
                        "name": (
                            f"adcde_{bind1_name}-x-{core_name}-{link2_name}-"
                            f"{bind2_name}"
                        ),
                        "bbs": (
                            stk.BuildingBlock(link2, [stk.BromoFactory()]),
                            stk.BuildingBlock(bind1, [stk.BromoFactory()]),
                            stk.BuildingBlock(bind2, [stk.BromoFactory()]),
                            stk.BuildingBlock(core, [stk.BromoFactory()]),
                        ),
                        "ru": "BADAC",
                    },
                    "abca": {
                        "name": (
                            f"abca_{bind1_name}-{link1_name}-{core_name}-x-x"
                        ),
                        "bbs": (
                            stk.BuildingBlock(link1, [stk.BromoFactory()]),
                            stk.BuildingBlock(bind1, [stk.BromoFactory()]),
                            stk.BuildingBlock(core, [stk.BromoFactory()]),
                        ),
                        "ru": "BACB",
                    },
                    "ebce": {
                        "name": (
                            f"ebce_x-{link1_name}-{core_name}-x-{bind2_name}"
                        ),
                        "bbs": (
                            stk.BuildingBlock(link1, [stk.BromoFactory()]),
                            stk.BuildingBlock(bind2, [stk.BromoFactory()]),
                            stk.BuildingBlock(core, [stk.BromoFactory()]),
                        ),
                        "ru": "BACB",
                    },
                    "adca": {
                        "name": (
                            f"adca_{bind1_name}-x-{core_name}-{link2_name}-x"
                        ),
                        "bbs": (
                            stk.BuildingBlock(link2, [stk.BromoFactory()]),
                            stk.BuildingBlock(bind1, [stk.BromoFactory()]),
                            stk.BuildingBlock(core, [stk.BromoFactory()]),
                        ),
                        "ru": "BACB",
                    },
                    "edce": {
                        "name": (
                            f"edce_x-x-{core_name}-{link2_name}-{bind2_name}"
                        ),
                        "bbs": (
                            stk.BuildingBlock(link2, [stk.BromoFactory()]),
                            stk.BuildingBlock(bind2, [stk.BromoFactory()]),
                            stk.BuildingBlock(core, [stk.BromoFactory()]),
                        ),
                        "ru": "BACB",
                    },
                    "abcda": {
                        "name": (
                            f"abcda_{bind1_name}-{link1_name}-{core_name}-"
                            f"{link2_name}-x"
                        ),
                        "bbs": (
                            stk.BuildingBlock(link1, [stk.BromoFactory()]),
                            stk.BuildingBlock(link2, [stk.BromoFactory()]),
                            stk.BuildingBlock(bind1, [stk.BromoFactory()]),
                            stk.BuildingBlock(core, [stk.BromoFactory()]),
                        ),
                        "ru": "CADBC",
                    },
                    "ebcde": {
                        "name": (
                            f"ebcde_x-{link1_name}-{core_name}-{link2_name}-"
                            f"{bind2_name}"
                        ),
                        "bbs": (
                            stk.BuildingBlock(link1, [stk.BromoFactory()]),
                            stk.BuildingBlock(link2, [stk.BromoFactory()]),
                            stk.BuildingBlock(bind2, [stk.BromoFactory()]),
                            stk.BuildingBlock(core, [stk.BromoFactory()]),
                        ),
                        "ru": "CADBC",
                    },
                }

                for option in options:
                    ligand_name = options[option]["name"]
                    ligand_name = normalise_names(ligand_name)

                    is_symmetric = symmetry_check(
                        composition=option,
                        building_blocks=options[option]["bbs"],
                        repeating_unit=options[option]["ru"],
                    )

                    if is_symmetric:

                        continue

                    build_ligand(
                        ligand_name=ligand_name,
                        building_blocks=options[option]["bbs"],
                        repeating_unit=options[option]["ru"],
                        ligand_dir=ligand_dir,
                        figures_dir=figures_dir,
                        ligand_db=ligand_db,
                        deduped_db=deduped_db,
                    )
                    count += 1


def explore_ligand(
    molecule: stk.Molecule,
    ligand_name: str,
    ligand_dir: pathlib.Path,
    ligand_db: atomlite.Database,
) -> None:
    """Do conformer scan."""
    st = time.time()
    conf_dir = ligand_dir / f"confs_{ligand_name}"
    conf_dir.mkdir(exist_ok=True)

    logging.info("building conformer ensemble of %s", ligand_name)

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
        "%s confs generated for %s in %s s",
        num_confs,
        ligand_name,
        round(time.time() - st, 2),
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
    components_dir = pathlib.Path(
        "/home/atarzia/workingspace/cpl/figures/components_2d"
    )
    ligand_dir.mkdir(exist_ok=True)
    calculation_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True, parents=True)
    components_dir.mkdir(exist_ok=True, parents=True)

    ligand_db = atomlite.Database(ligand_dir / "ligands.db")
    deduped_db = atomlite.Database(ligand_dir / "deduped_ligands.db")

    # Generate all ligands from core, binder and connector pools.
    generate_all_ligands(
        ligand_dir=ligand_dir,
        figures_dir=figures_dir,
        components_dir=components_dir,
        ligand_db=ligand_db,
        deduped_db=deduped_db,
    )
    logging.info("built %s ligands", ligand_db.num_entries())
    logging.info("built %s deduped ligands", deduped_db.num_entries())
    logging.info(
        "%s ligands in key database",
        atomlite.Database(ligand_dir / "keys.db").num_entries() / 3,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    main()
