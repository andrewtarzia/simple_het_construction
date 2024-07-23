

def main():
    args = _parse_args()
    ligand_dir = pathlib.Path("/home/atarzia/workingspace/cpl/case_study_1")
    calculation_dir = pathlib.Path("/home/atarzia/workingspace/cpl/cs1_calculations")
    figures_dir = pathlib.Path("/home/atarzia/workingspace/cpl/figures/cs1/")
    ligand_dir.mkdir(exist_ok=True)
    calculation_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True, parents=True)

    ligand_db = atomlite.Database(ligand_dir / "case_study_1.db")
    pair_db = atomlite.Database(ligand_dir / "cs1_pairs.db")

    # Build all ligands.
    ligand_smiles = {
        # Converging.
        "lab_0": "C1=CC=NC=C1C1=CC(C#CC2=CC(C(NC3=CN=CC=C3)=O)=CC=C2)=CC=C1",
        # Diverging.
        "la_0": ("C1=NC=C2C=CC=C(C#CC3=CC=CN=C3)C2=C1"),
        "lb_0": (
            "C([H])1C([H])=C2C(C([H])=C([H])C([H])=C2C2=C([H])C([H])=C(C3C([H]"
            ")=NC([H])=C([H])C=3[H])C([H])=C2[H])=C([H])N=1"
        ),
        "lc_0": ("C1=CC(=CN=C1)C#CC2=CN=CC=C2"),
        "ld_0": ("C1=CC(=CN=C1)C2=CC=C(C=C2)C3=CN=CC=C3"),
    }
    for lname in ligand_smiles:
        lowe_file = ligand_dir / f"{lname}_lowe.mol"
        if lowe_file.exists():
            molecule = stk.BuildingBlock.init_from_file(lowe_file)
        else:
            # Build polymer.
            molecule = stk.BuildingBlock(smiles=ligand_smiles[lname])
            molecule = stko.ETKDG().optimize(molecule)

        if not ligand_db.has_entry(key=lname):
            explore_ligand(
                molecule=molecule,
                ligand_name=lname,
                ligand_dir=ligand_dir,
                figures_dir=figures_dir,
                calculation_dir=calculation_dir,
                ligand_db=ligand_db,
            )

        if args.plot_ligands:
            plot_ligand(
                ligand_name=lname,
                ligand_db=ligand_db,
                ligand_dir=ligand_dir,
                figures_dir=figures_dir,
            )

    if args.plot_ligands:
        plot_flexes(ligand_db=ligand_db, figures_dir=figures_dir)

    for ligand1, ligand2 in it.combinations(ligand_smiles, 2):
        logging.info(f"analysing {ligand1} and {ligand2}")
        key = f"{ligand1}_{ligand2}"
        if pair_db.has_property_entry(key):
            continue

        analyse_ligand_pair(
            ligand1=ligand1,
            ligand2=ligand2,
            key=key,
            ligand_db=ligand_db,
            pair_db=pair_db,
        )

        raise SystemExit


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    main()
