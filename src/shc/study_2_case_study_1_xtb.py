"""Script to build the ligands in this case study."""

import argparse
import itertools as it
import json
import logging
import pathlib
import time

import atomlite
import bbprep
import matplotlib.pyplot as plt
import numpy as np
import stk
import stko
from rdkit.Chem import AllChem as rdkit  # noqa: N813
from rdkit.Chem import rdMolDescriptors, rdmolops, rdMolTransforms

from shc.definitions import EnvVariables, Study1EnvVariables
from shc.matching_functions import (
    angle_test,
    mismatch_test,
    plot_pair_position,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    return parser.parse_args()


def analyse_ligand_pair(  # noqa: PLR0913
    ligand1: str,
    ligand2: str,
    key: str,
    ligand_db: atomlite.Database,
    pair_db: atomlite.Database,
    figures_dir: pathlib.Path,
) -> None:
    """Analyse a pair of ligands."""
    raise SystemExit("get low e conf, and rxn conf")

    raise SystemExit("build macrocycle")

    raise SystemExit("pre opt")

    raise SystemExit("do gfnff with the constrains")

def get_lowest_energy_xtb_conformer(
    molecule: stk.Molecule,
    name: str,
    charge: int,
    calc_dir: pathlib.Path,
    ligand_db: atomlite.Database,
) -> None:
    """Use ligand ensemble and xtb to get lowest energy conformer."""
    low_energy_structure = calc_dir / f"{name}_lowe_xtbff.mol"
    rxn_structure = calc_dir / f"{name}_rxn.mol"

    generator = bbprep.generators.ETKDG(num_confs=100)
    ensemble = generator.generate_conformers(molecule)
    min_energy = float("inf")
    min_conformer = None
    min_binder_binder_angle = float("inf")
    rxn_conformer = None
    for conformer in ensemble.yield_conformers():
        conf_name = f"{name}_{conformer.conformer_id}"
        xtbffopt_output = calc_dir / f"{conf_name}_xtbff.mol"
        if not xtbffopt_output.exists():
            output_dir = calc_dir / f"{conf_name}_xtbffopt"
            logging.info("    GFN-FF optimisation of %s", conf_name)
            xtbff_opt = stko.XTBFF(
                xtb_path=EnvVariables.xtb_path,
                output_dir=output_dir,
                num_cores=6,
                charge=charge,
                opt_level="normal",
            )
            xtbffopt_mol = xtbff_opt.optimize(mol=conformer.molecule)
            xtbffopt_mol.write(xtbffopt_output)
        else:
            logging.info("    loading %s", xtbffopt_output)
            xtbffopt_mol = conformer.molecule.with_structure_from_file(
                str(xtbffopt_output)
            )

        energy = calculate_xtb_energy(
            molecule=xtbffopt_mol,
            name=conf_name,
            charge=charge,
            calc_dir=calc_dir,
        )
        if energy < min_energy:
            min_energy = energy
            min_conformer = bbprep.Conformer(
                molecule=xtbffopt_mol,
                conformer_id=conformer.conformer_id,
                source=conformer.source,
                permutation=None,
            )

        analyser = stko.molecule_analysis.DitopicThreeSiteAnalyser()
        binder_binder_angle = analyser.get_binder_binder_angle(
            molecule=bbprep.FurthestFGs().modify(
                building_block=stk.BuildingBlock.init_from_molecule(
                    molecule=xtbffopt_mol,
                    functional_groups=(stko.functional_groups.CNCFactory(),),
                ),
                desired_functional_groups=2,
            ),
        )
        if binder_binder_angle < min_binder_binder_angle:
            min_binder_binder_angle = binder_binder_angle
            rxn_conformer = bbprep.Conformer(
                molecule=xtbffopt_mol,
                conformer_id=conformer.conformer_id,
                source=conformer.source,
                permutation=None,
            )

    logging.info(
        "lowest energy for %s is %s",
        name,
        min_conformer.conformer_id,
    )
    min_conformer.molecule.write(low_energy_structure)
    rxn_conformer.molecule.write(rxn_structure)
    entry = atomlite.Entry.from_rdkit(
        key=name,
        molecule=rxn_conformer.molecule.to_rdkit_mol(),
        properties={"energy;au": min_energy},
    )
    ligand_db.add_entries(entry)


def main() -> None:
    """Run script."""
    ligand_dir = pathlib.Path("/home/atarzia/workingspace/cpl/cs1xtb_ligands")
    calculation_dir = pathlib.Path(
        "/home/atarzia/workingspace/cpl/cs1xtb_calculations"
    )
    figures_dir = pathlib.Path(
        "/home/atarzia/workingspace/cpl/figures/cs1xtb/"
    )
    ligand_dir.mkdir(exist_ok=True)
    calculation_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True, parents=True)

    ligand_db = atomlite.Database(ligand_dir / "cs1xtb_ligands.db")
    pair_db = atomlite.Database(ligand_dir / "cs1xtb_pairs.db")

    # Build all ligands.
    ligand_smiles = {
        ##### Converging. #####
        # From Chand.
        "lab_0": "C1=CC=NC=C1C1=CC(C#CC2=CC(C(NC3=CN=CC=C3)=O)=CC=C2)=CC=C1",
        # From molinksa.
        "m2h_0": "C1=CC(=CC(=C1)C#CC2=CN=CC=C2)C#CC3=CN=CC=C3",
        # From study 1.
        "sl1_0": "C1=NC=CC(C2=CC=C3OC4C=CC(C5C=CN=CC=5)=CC=4C3=C2)=C1",
        "sl2_0": "C1=CC(=CC(=C1)C2=CC=NC=C2)C3=CC=NC=C3",
        "sl3_0": "C1=CN=CC=C1C2=CC=C(S2)C3=CC=NC=C3",
        ##### Diverging. #####
        # From Chand.
        "la_0": ("C1=NC=C2C=CC=C(C#CC3=CC=CN=C3)C2=C1"),
        "lb_0": (
            "C([H])1C([H])=C2C(C([H])=C([H])C([H])=C2C2=C([H])C([H])=C(C3C([H]"
            ")=NC([H])=C([H])C=3[H])C([H])=C2[H])=C([H])N=1"
        ),
        "lc_0": ("C1=CC(=CN=C1)C#CC2=CN=CC=C2"),
        "ld_0": ("C1=CC(=CN=C1)C2=CC=C(C=C2)C3=CN=CC=C3"),
        # From molinksa.
        "m4q_0": "C1=CC=C2C(=C1)C=C(C=N2)C#CC3=CN=CC=C3",
        "m4p_0": "C1=CN=CC(C#CC2=CN=C(C)C=C2)=C1",
        # From study 1.
        "sla_0": (
            "C1=CN=CC2C(C3=CC=C(C#CC4=CC5C6C=C(C#CC7=CC=C(C8=CC=CC9C=C"
            "N=CC8=9)C=C7)C=CC=6OC=5C=C4)C=C3)=CC=CC1=2"
        ),
        "slb_0": (
            "C1=CN=CC2C(C3=CC=C(C#CC4N=C(C#CC5=CC=C(C6=CC=CC7C=CN=CC6="
            "7)C=C5)C=CC=4)C=C3)=CC=CC1=2"
        ),
        "slc_0": (
            "C1C2=C(C(=CC=C2)C2C=CC(C#CC3=CC=CC(C#CC4C=CC(C5C6=C(C=CN="
            "C6)C=CC=5)=CC=4)=C3)=CC=2)C=NC=1"
        ),
        "sld_0": (
            "C1C2=C(C(=CC=C2)C2C=CC(C#CC3=CC=C(C#CC4C=CC(C5C6=C(C=CN=C"
            "6)C=CC=5)=CC=4)S3)=CC=2)C=NC=1"
        ),
        # Experimental.
        "e10_0": (
            "C1=CC(C#CC2=CC3C4C=C(C#CC5=CC=CN=C5)C=CC=4N(C)C=3C=C2)=CN=C1"
        ),
        "e11_0": "C1N=CC=CC=1C1=CC2=C(C3=C(C2(C)C)C=C(C2=CN=CC=C2)C=C3)C=C1",
        "e12_0": "C1=CC=C(C2=CC3C(=O)C4C=C(C5=CN=CC=C5)C=CC=4C=3C=C2)C=N1",
        "e13_0": (
            "C1C=C(N2C(=O)C3=C(C=C4C(=C3)C3(C5=C(C4(C)CC3)C=C3C(C(N(C3="
            "O)C3C=CC=NC=3)=O)=C5)C)C2=O)C=NC=1"
        ),
        "e14_0": (
            "C1=CN=CC(C#CC2C=CC3C(=O)C4C=CC(C#CC5=CC=CN=C5)=CC=4C=3C=2)=C1"
        ),
        "e16_0": (
            "C(C1=CC2C3C=C(C4=CC=NC=C4)C=CC=3C(OC)=C(OC)C=2C=C1)1=CC=NC=C1"
        ),
        "e17_0": (
            "C12C=CN=CC=1C(C#CC1=CC=C3C(C(C4=C(N3C)C=CC(C#CC3=CC=CC5C3="
            "CN=CC=5)=C4)=O)=C1)=CC=C2"
        ),
        "e18_0": (
            "C1(=CC=NC=C1)C#CC1=CC2C3C=C(C#CC4=CC=NC=C4)C=CC=3C(OC)=C(O"
            "C)C=2C=C1"
        ),
    }

    for lname in ligand_smiles:
        if not ligand_db.has_entry(key=lname):
            # Build polymer.
            molecule = stk.BuildingBlock(smiles=ligand_smiles[lname])
            molecule = stko.ETKDG().optimize(molecule)
            get_lowest_energy_xtb_conformer(
                molecule=molecule,
                name=lname,
                charge=0,
                calc_dir=calculation_dir,
                ligand_db=ligand_db,
            )

    targets = (
        ("sla_0", "sl1_0"),
        ("slb_0", "sl1_0"),
        ("slc_0", "sl1_0"),
        ("sld_0", "sl1_0"),
        ("sla_0", "sl2_0"),
        ("slb_0", "sl2_0"),
        ("slc_0", "sl2_0"),
        ("sld_0", "sl2_0"),
        ("lab_0", "la_0"),
        ("lab_0", "lb_0"),
        ("lab_0", "lc_0"),
        ("lab_0", "ld_0"),
        ("m2h_0", "m4q_0"),
        ("m2h_0", "m4p_0"),
        ("e16_0", "e10_0"),
        ("e16_0", "e17_0"),
        ("e10_0", "e17_0"),
        ("e11_0", "e10_0"),
        ("e16_0", "e14_0"),
        ("e18_0", "e14_0"),
        ("e18_0", "e10_0"),
        ("e12_0", "e10_0"),
        ("e11_0", "e14_0"),
        ("e12_0", "e14_0"),
        ("e11_0", "e13_0"),
        ("e12_0", "e13_0"),
        ("e13_0", "e14_0"),
        ("e11_0", "e12_0"),
    )

    for ligand1, ligand2 in targets:
        key = f"{ligand1}_{ligand2}"

        if pair_db.has_property_entry(key):
            continue

        logging.info("analysing %s and %s", ligand1, ligand2)
        analyse_ligand_pair(
            ligand1=ligand1,
            ligand2=ligand2,
            key=key,
            ligand_db=ligand_db,
            pair_db=pair_db,
            figures_dir=figures_dir,
        )
        raise SystemExit


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    main()
