import itertools as it
import json
import pathlib
import shutil

import numpy as np
import stk

from shc.definitions import MatchingSettings, Study1EnvVariables
from shc.scripts.study_1.ligand_analysis import (
    conformer_generation_uff,
    study_1_get_test_1,
    study_1_get_test_2,
)
from shc.study_1 import AromaticCNCFactory

from .case_data import CaseData


def test_study_1(case_data: CaseData) -> None:  # noqa: C901, PLR0915, PLR0912
    _output_dir = pathlib.Path(__file__).resolve().parents[0] / "test_output"
    saved_output_dir = (
        pathlib.Path(__file__).resolve().parents[0] / "saved_output"
    )
    position_matrices = (
        pathlib.Path(__file__).resolve().parents[0] / "position_matrices"
    )
    if _output_dir.exists():
        shutil.rmtree(_output_dir)

    _output_dir.mkdir(exist_ok=False)
    saved_output_dir.mkdir(exist_ok=True)
    position_matrices.mkdir(exist_ok=True)

    lsmiles = {
        case_data.l1_name: case_data.l1_smiles,
        case_data.l2_name: case_data.l2_smiles,
    }
    for lig, smi in lsmiles.items():
        unopt_mol = stk.BuildingBlock(
            smiles=smi,
            functional_groups=(AromaticCNCFactory(),),
        )
        conformer_generation_uff(
            molecule=unopt_mol,
            name=lig,
            lowe_output=_output_dir / f"{lig}_lowe.mol",
            conf_data_file=_output_dir / f"{lig}_conf_uff_data.json",
        )
        shutil.copy(
            _output_dir / f"{lig}_conf_uff_data.json",
            saved_output_dir / f"{lig}_conf_uff_data.json",
        )

    conf_data_suffix = "conf_uff_data"
    structure_results = {}
    for ligand in lsmiles:
        structure_results[ligand] = {}
        conf_data_file = _output_dir / f"{ligand}_{conf_data_suffix}.json"
        with conf_data_file.open() as f:
            property_dict = json.load(f)

        for cid in property_dict:
            pdi = property_dict[cid]["NN_BCN_angles"]
            # 180 - angle, to make it the angle toward the binding
            # interaction. Minus 90  to convert to the bite-angle.
            ba = ((180 - pdi["NN_BCN1"]) - 90) + ((180 - pdi["NN_BCN2"]) - 90)
            property_dict[cid]["bite_angle"] = ba

            if cid in ("0", "22", "89", "10", "34"):
                saved_file = position_matrices / f"{ligand}_c{cid}_cuff.npy"
                conf_file = _output_dir / f"{ligand}_c{cid}_cuff.mol"
                conf_mol = stk.BuildingBlock.init_from_file(conf_file)
                if not saved_file.exists():
                    np.save(
                        file=saved_file,
                        arr=conf_mol.get_position_matrix(),
                        allow_pickle=False,
                    )
                assert np.allclose(
                    a=conf_mol.get_position_matrix(),
                    b=conf_mol.with_position_matrix(
                        np.load(saved_file)
                    ).get_position_matrix(),
                    atol=1e-6,
                )

        structure_results[ligand] = property_dict

    # Define minimum energies for all ligands.
    low_energy_values = {}
    for ligand, sres in structure_results.items():
        min_energy = 1e24
        min_e_cid = 0
        for cid in sres:
            energy = sres[cid]["UFFEnergy;kj/mol"]
            if energy < min_energy:
                min_energy = energy
                min_e_cid = cid
        low_energy_values[ligand] = (min_e_cid, min_energy)

    assert low_energy_values[ligand][0] == case_data.low_energies[ligand][0]
    assert np.isclose(
        low_energy_values[ligand][1], case_data.low_energies[ligand][1]
    )

    ligand_pairings = [(case_data.l1_name, case_data.l2_name)]

    pair_info = {}
    min_geom_scores = {}
    for small_l, large_l in ligand_pairings:
        min_geom_score = 1e24
        pair_name = f"{small_l},{large_l}"
        pair_info[pair_name] = {}
        small_l_dict = structure_results[small_l]
        large_l_dict = structure_results[large_l]

        # Iterate over the product of all conformers.
        for small_cid, large_cid in it.product(small_l_dict, large_l_dict):
            cid_name = f"{small_cid},{large_cid}"
            # Calculate geom score for both sides together.
            large_c_dict = large_l_dict[large_cid]
            small_c_dict = small_l_dict[small_cid]

            # Calculate final geometrical properties.
            # T1.
            angle_dev = study_1_get_test_1(
                large_c_dict=large_c_dict,
                small_c_dict=small_c_dict,
            )
            # T2.
            length_dev = study_1_get_test_2(
                large_c_dict=large_c_dict,
                small_c_dict=small_c_dict,
                pdn_distance=MatchingSettings.vector_length,
            )
            geom_score = abs(angle_dev - 1) + abs(length_dev - 1)

            small_energy = small_l_dict[small_cid]["UFFEnergy;kj/mol"]
            small_strain = small_energy - low_energy_values[small_l][1]
            large_energy = large_l_dict[large_cid]["UFFEnergy;kj/mol"]
            large_strain = large_energy - low_energy_values[large_l][1]
            if (
                small_strain > Study1EnvVariables.strain_cutoff
                or large_strain > Study1EnvVariables.strain_cutoff
            ):
                continue

            min_geom_score = min((geom_score, min_geom_score))
            pair_info[pair_name][cid_name] = {
                "geom_score": geom_score,
                "large_dihedral": large_c_dict["NCCN_dihedral"],
                "small_dihedral": small_c_dict["NCCN_dihedral"],
                "angle_deviation": angle_dev,
                "length_deviation": length_dev,
            }
        min_geom_scores[pair_name] = round(min_geom_score, 2)

    for (key, value), (tkey, tvalue) in zip(
        case_data.min_gs.items(), min_geom_scores.items(), strict=False
    ):
        assert key == tkey
        assert np.isclose(value, tvalue)

    for rdict in pair_info.values():
        geom_scores = []
        for cid_pair in rdict:
            if (
                abs(rdict[cid_pair]["large_dihedral"])
                > Study1EnvVariables.dihedral_cutoff
                or abs(rdict[cid_pair]["small_dihedral"])
                > Study1EnvVariables.dihedral_cutoff
            ):
                continue

            geom_scores.append(rdict[cid_pair]["geom_score"])

        mean = np.mean(geom_scores)
        std = np.std(geom_scores)

    assert np.isclose(mean, case_data.mean_g, atol=1e-2)
    assert np.isclose(std, case_data.std_g, atol=1e-2)
