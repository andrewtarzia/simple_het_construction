import itertools as it
import pathlib

import atomlite
import numpy as np
import stk
import stko

from shc.definitions import EnvVariables
from shc.matching_functions import mismatch_test
from shc.study_2 import explore_ligand

from .case_data import CaseData


def test_study_2(case_data: CaseData) -> None:
    saved_output_dir = (
        pathlib.Path(__file__).resolve().parents[0] / "saved_output"
    )
    saved_output_dir.mkdir(exist_ok=True)
    ligand_db = atomlite.Database(saved_output_dir / "test_ligands.db")
    for lname, lsmiles in {
        case_data.l1_name: case_data.l1_smiles,
        case_data.l2_name: case_data.l2_smiles,
    }.items():
        lowe_file = saved_output_dir / f"{lname}_lowe.mol"
        if lowe_file.exists():
            molecule = stk.BuildingBlock.init_from_file(lowe_file)
        else:
            # Build polymer.
            molecule = stk.BuildingBlock(smiles=lsmiles)
            molecule = stko.ETKDG().optimize(molecule)

        if not ligand_db.has_entry(key=lname):
            explore_ligand(
                molecule=molecule,
                ligand_name=lname,
                ligand_dir=saved_output_dir,
                ligand_db=ligand_db,
            )

    ligand1_entry = ligand_db.get_entry(case_data.l1_name)
    ligand1_confs = ligand1_entry.properties["conf_data"]
    ligand2_entry = ligand_db.get_entry(case_data.l2_name)
    ligand2_confs = ligand2_entry.properties["conf_data"]
    s1s = []
    s2s = []
    # Iterate over the product of all conformers.
    for cid_1, cid_2 in it.product(ligand1_confs, ligand2_confs):
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
            continue

        # Check torsion.
        torsion1 = abs(ligand1_confs[cid_1]["NCCN_dihedral"])
        torsion2 = abs(ligand2_confs[cid_2]["NCCN_dihedral"])
        if (
            torsion1 > EnvVariables.dihedral_cutoff
            or torsion2 > EnvVariables.dihedral_cutoff
        ):
            continue

        # Calculate geom score for both sides together.
        c_dict1 = ligand1_confs[cid_1]
        c_dict2 = ligand2_confs[cid_2]

        # Calculate final geometrical properties.
        pair_results = mismatch_test(
            c_dict1=c_dict1,
            c_dict2=c_dict2,
            k_angle=EnvVariables.k_angle,
            k_bond=EnvVariables.k_bond,
        )

        s1s.append(float(pair_results.state_1_result))
        s2s.append(float(pair_results.state_2_result))

    assert len(s1s) == case_data.len_s1
    assert len(s2s) == case_data.len_s2
    assert np.isclose(np.mean(s1s), case_data.mean_s1, atol=1e-6)
    assert np.isclose(np.mean(s2s), case_data.mean_s2, atol=1e-6)
    assert np.isclose(np.min(s1s), case_data.min_s1, atol=1e-6)
    assert np.isclose(np.min(s2s), case_data.min_s2, atol=1e-6)
