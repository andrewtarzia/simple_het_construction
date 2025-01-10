"""Script to analyse all cages constructed."""

import glob
import json
import logging
import os

import stk
from pywindow_module import PyWindow
from study_1_cage_plotting import plot_strain_pore_sasa
from topologies import heteroleptic_cages, ligand_cage_topologies
from utilities import (
    AromaticCNCFactory,
    get_furthest_pair_FGs,
    get_mm_distance,
    get_order_values,
    get_pore_angle,
    get_stab_energy,
    get_xtb_energy,
    get_xtb_enthalpy,
    get_xtb_free_energy,
    get_xtb_gsasa,
    get_xtb_gsolv,
    get_xtb_sasa,
    get_xtb_strain,
)


def get_min_order_parameter(molecule):
    order_results = get_order_values(mol=molecule, metal=46)
    return order_results["sq_plan"]["min"]


def main() -> None:  # noqa: C901, PLR0912, PLR0915
    """Run script."""
    li_path = liga_path()
    ligands = {
        i.split("/")[-1].replace("_opt.mol", ""): (
            stk.BuildingBlock.init_from_file(
                path=i,
                functional_groups=(AromaticCNCFactory(),),
            )
        )
        for i in glob.glob(str(li_path / "*_opt.mol"))
    }

    ligands = {
        i: ligands[i].with_functional_groups(
            functional_groups=get_furthest_pair_FGs(ligands[i])
        )
        for i in ligands
    }

    _wd = cage_path()
    _cd = calc_path()
    _ld = liga_path()
    _pd = project_path()

    property_dictionary = {
        "cis": {
            "charge": 4,
            "exp_lig": 2,
        },
        "trans": {
            "charge": 4,
            "exp_lig": 2,
        },
        "m2": {
            "charge": 4,
            "exp_lig": 1,
        },
        "m3": {
            "charge": 6,
            "exp_lig": 1,
        },
        "m4": {
            "charge": 8,
            "exp_lig": 1,
        },
        "m6": {
            "charge": 12,
            "exp_lig": 1,
        },
        "m12": {
            "charge": 24,
            "exp_lig": 1,
        },
        "m24": {
            "charge": 48,
            "exp_lig": 1,
        },
        "m30": {
            "charge": 60,
            "exp_lig": 1,
        },
    }

    structure_files = []
    hets = heteroleptic_cages()
    lct = ligand_cage_topologies()
    for ligname in lct:
        toptions = lct[ligname]
        for topt in toptions:
            sname = _wd / f"{topt}_{ligname}_opt.mol"
            structure_files.append(str(sname))

    for l1, l2 in hets:
        for topt in ("cis", "trans"):
            sname = _wd / f"{topt}_{l1}_{l2}_opt.mol"
            structure_files.append(str(sname))
    logging.info(f"there are {len(structure_files)} structures.")

    structure_results = {
        i.split("/")[-1].replace("_opt.mol", ""): {} for i in structure_files
    }
    structure_res_file = os.path.join(_wd, "all_structure_res.json")
    if os.path.exists(structure_res_file):
        with open(structure_res_file) as f:
            structure_results = json.load(f)
    else:
        for s_file in structure_files:
            name = s_file.split("/")[-1].replace("_opt.mol", "")
            splits = name.split("_")
            if len(splits) == 2:
                prefix, lname = splits
                if prefix not in ligand_cage_topologies()[lname]:
                    continue
            else:
                prefix = splits[0]
                lname = None

            properties = property_dictionary[prefix]
            charge = properties["charge"]
            exp_lig = properties["exp_lig"]
            molecule = stk.BuildingBlock.init_from_file(s_file)

            structure_results[name]["xtb_solv_opt_gasenergy_au"] = (
                get_xtb_energy(
                    molecule=molecule,
                    name=name,
                    charge=charge,
                    calc_dir=_cd,
                    solvent=None,
                )
            )
            structure_results[name]["xtb_solv_opt_dmsoenergy_au"] = (
                get_xtb_energy(
                    molecule=molecule,
                    name=name,
                    charge=charge,
                    calc_dir=_cd,
                    solvent="dmso",
                )
            )
            structure_results[name]["xtb_sasa"] = get_xtb_sasa(
                molecule=molecule,
                name=name,
                charge=charge,
                calc_dir=_cd,
                solvent="dmso",
            )

            if name in (
                "m2_la",
                "m2_lb",
                "m2_lc",
                "m2_ld",
                "m2_l1",
                "m3_l1",
                "m4_l1",
                "m6_l1",
                "m12_l1",
                "m12_l2",
                "cis_l1_la",
                "cis_l1_lb",
                "cis_l1_lc",
                "cis_l1_ld",
                "cis_l2_la",
                "cis_l2_lb",
                "cis_l2_lc",
                "cis_l2_ld",
            ):
                structure_results[name]["xtb_solv_opt_dmsofreeenergy_au"] = (
                    get_xtb_free_energy(
                        molecule=molecule,
                        name=name,
                        charge=charge,
                        calc_dir=_cd,
                        solvent="dmso",
                    )
                )
                structure_results[name]["xtb_solv_opt_dmsoenthalpy_au"] = (
                    get_xtb_enthalpy(
                        molecule=molecule,
                        name=name,
                        charge=charge,
                        calc_dir=_cd,
                        solvent="dmso",
                    )
                )

            structure_results[name]["stabilisation_energy"] = get_stab_energy(
                molecule=molecule,
                name=name,
                charge=charge,
                calc_dir=_cd,
                solvent="gas",
                metal_atom_nos=(46,),
                cage_energy=structure_results[name][
                    "xtb_solv_opt_gasenergy_au"
                ],
            )
            structure_results[name]["E1'"] = (
                structure_results[name]["xtb_solv_opt_gasenergy_au"]
                - structure_results[name]["stabilisation_energy"]
            )
            structure_results[name]["xtb_gsolv_au"] = get_xtb_gsolv(
                name=name,
                calc_dir=_cd,
                solvent="dmso",
            )
            structure_results[name]["xtb_gsasa_au"] = get_xtb_gsasa(
                name=name,
                calc_dir=_cd,
                solvent="dmso",
            )

            if prefix in ("cis", "trans", "m2"):
                structure_results[name]["pore_angle"] = get_pore_angle(
                    molecule=molecule,
                    metal_atom_num=46,
                )
                structure_results[name]["mm_distance"] = get_mm_distance(
                    molecule=molecule,
                    metal_atom_num=46,
                )

            structure_results[name]["xtb_lig_strain_au"] = get_xtb_strain(
                molecule=molecule,
                name=name,
                liga_dir=_ld,
                calc_dir=_cd,
                exp_lig=exp_lig,
                solvent="dmso",
            )

            min_order_param = get_min_order_parameter(molecule)
            structure_results[name]["min_order_param"] = min_order_param

            structure_results[name]["pw_results"] = PyWindow(
                name, _cd
            ).get_results(molecule)

        with open(structure_res_file, "w") as f:
            json.dump(structure_results, f, indent=4)

    for name in structure_results:
        if "pw_results" in structure_results[name]:
            print(
                name, structure_results[name]["pw_results"]["pore_volume_opt"]
            )

    plot_strain_pore_sasa(
        results_dict=structure_results,
        outname="strain_pore_sasa",
    )
    plotting.plot_strain(
        results_dict=structure_results,
        outname="xtb_strain_energy",
        yproperty="xtb_lig_strain_au",
    )
    plotting.plot_sasa(
        results_dict=structure_results,
        outname="xtb_sasa",
        yproperty="xtb_sasa",
    )
    plotting.plot_pore(
        results_dict=structure_results,
        outname="cage_pw_diameter",
        yproperty="pore_diameter_opt",
    )
    raise SystemExit
    plotting.plot_all_contributions(
        results_dict=structure_results,
        outname="main_contributions",
    )
    plotting.plot_gsasa(
        results_dict=structure_results,
        outname="main_ssolv",
        yproperty="xtb_gsasa_au",
    )
    plotting.plot_gsolv(
        results_dict=structure_results,
        outname="main_gsolv",
        yproperty="xtb_gsolv_au",
    )
    plotting.plot_topo_energy(
        results_dict=structure_results,
        outname="main_topology_ey",
    )

    plotting.plot_stab_energy(
        results_dict=structure_results,
        outname="stabilisation_ey",
    )
    plotting.plot_topo_energy(
        results_dict=structure_results,
        outname="gas_topology_ey",
        solvent="gas",
    )
    plotting.plot_qsqp(
        results_dict=structure_results,
        outname="cage_qsqp",
        yproperty="min_order_param",
    )
    plotting.plot_property(
        results_dict=structure_results,
        outname="cage_poreangle",
        yproperty="pore_angle",
    )
    plotting.plot_property(
        results_dict=structure_results,
        outname="cage_mm_distance",
        yproperty="mm_distance",
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    main()
