"""Script to build the ligand in this project."""

import itertools as it
import logging
import pathlib

import atomlite
import stk
from rdkit.Chem import AllChem as rdkit  # noqa: N813
from rdkit.Chem import Draw

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
    # 12: "C1=CC2=C(C=C(C=C2)Br)C3=C1C=CC(=C3)Br",
    13: "CN1C2=C(C=C(C=C2)Br)C(=O)C3=C1C=CC(=C3)Br",
    14: "C1=CC=C(C(=C1)Br)Br",
    15: "C1=C(N=NN1Br)Br",
}
linker_smiles = {
    0: "C1=CC(=CC=C1Br)Br",
    1: "Brc1cccc(Br)c1",
    2: "BrC#CBr",
    # 3: "C1=CC=C2C(=C1)C(=C3C=CC=CC3=C2Br)Br",
    4: "C1=CC=C(C(=C1)Br)Br",
    5: "C1=C(N=NN1Br)Br",
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
    # 8: "C1=CC=C2C(=C1)C=NC=C2Br",
    # 9: "C1=CC=C2C(=C1)C=C(C=N2)Br",
    # "C1=CC=C2C(=C1)C(=C3C=CC=CC3=N2)Br",  # noqa: ERA001
}


def generate_all_ligands(  # noqa: C901
    ligand_dir: pathlib.Path,
    figures_dir: pathlib.Path,
    components_dir: pathlib.Path,
    deduped_db: atomlite.Database,
) -> None:
    """Iterate and generate through all components."""
    # Start a fresh keys db for deduping.
    if (ligand_dir / "keys.db").exists():
        (ligand_dir / "keys.db").unlink()
    key_db = atomlite.Database(ligand_dir / "keys.db")

    for lig, smi in core_smiles.items():
        rdkit_mol = rdkit.MolFromSmiles(smi)
        Draw.MolToFile(
            rdkit_mol, components_dir / f"core_{lig}_2d.png", size=(300, 300)
        )

    for lig, smi in linker_smiles.items():
        rdkit_mol = rdkit.MolFromSmiles(smi)
        Draw.MolToFile(
            rdkit_mol, components_dir / f"link_{lig}_2d.png", size=(300, 300)
        )

    for lig, smi in binder_smiles.items():
        rdkit_mol = rdkit.MolFromSmiles(smi)
        Draw.MolToFile(
            rdkit_mol, components_dir / f"bind_{lig}_2d.png", size=(300, 300)
        )

    count = 0

    for core_name, core in core_smiles.items():
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
                        if deduped_db.has_entry(ligand_name):
                            deduped_db.remove_entries(ligand_name)
                            logging.info(
                                "deleting %s because symmetric", ligand_name
                            )
                        continue

                    build_ligand(
                        ligand_name=ligand_name,
                        building_blocks=options[option]["bbs"],
                        repeating_unit=options[option]["ru"],
                        ligand_dir=ligand_dir,
                        figures_dir=figures_dir,
                        deduped_db=deduped_db,
                        key_db=key_db,
                    )
                    count += 1


def main() -> None:
    """Run script."""
    raise SystemExit(
        "rerun component grid making - components_2d, delete images, and use rdkit to make a nice grid"
    )
    raise SystemExit("rerun build ligands to not use RMSD weirdly")
    raise SystemExit("rerun with new conformer db file")
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

    deduped_db = atomlite.Database(ligand_dir / "deduped_ligands.db")

    # Generate all ligands from core, binder and connector pools.
    generate_all_ligands(
        ligand_dir=ligand_dir,
        figures_dir=figures_dir,
        components_dir=components_dir,
        deduped_db=deduped_db,
    )

    logging.info("built %s ligands", deduped_db.num_entries())

    logging.info("cleaning up for removed components")
    rmed_cores = (5, 6, 7, 12)
    rmed_linkers = (3,)
    rmed_binders = (5, 8, 9)
    entries_to_delete = []
    for entry in deduped_db.get_entries():
        comp_split = entry.properties["composition_pattern"].split("-")
        b1, l1, c, l2, b2 = comp_split
        if (
            (
                (b1 != "x" and int(b1) in rmed_binders)
                or (b2 != "x" and int(b2) in rmed_binders)
            )
            or (l1 != "x" and int(l1) in rmed_linkers)
            or (
                (l2 != "x" and int(l2) in rmed_linkers)
                or (c != "x" and int(c) in rmed_cores)
            )
        ):
            entries_to_delete.append(entry.key)

    logging.info("removing %s ligands", len(entries_to_delete))
    deduped_db.remove_entries(keys=entries_to_delete)
    logging.info("built %s deduped ligands", deduped_db.num_entries())


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    main()
