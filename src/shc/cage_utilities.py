"""Module for utility functions."""

import json
import logging
import os
import re

import numpy as np
import pymatgen.core as pmg
import stk
import stko
from pymatgen.analysis.local_env import LocalStructOrderParams
from scipy.spatial.distance import euclidean


def convert_stk_to_pymatgen(stk_mol):
    """Convert stk.Molecule to pymatgen.Molecule.

    Parameters
    ----------
    stk_mol : :class:`stk.Molecule`
        Stk molecule to convert.

    Returns:
    -------
    pmg_mol : :class:`pymatgen.Molecule`
        Corresponding pymatgen Molecule.

    """
    stk_mol.write("temp.xyz")
    pmg_mol = pmg.Molecule.from_file("temp.xyz")
    os.system("rm temp.xyz")

    return pmg_mol


def calculate_sites_order_values(
    molecule,
    site_idxs,
    target_species_type=None,
    neigh_idxs=None,
):
    """Calculate order parameters around metal centres.

    Parameters
    ----------
    molecule : :class:`pmg.Molecule` or :class:`pmg.Structure`
        Pymatgen (pmg) molecule/structure to analyse.

    site_idxs : :class:`list` of :class:`int`
        Atom ids of sites to calculate OP of.

    target_species_type : :class:`str`
        Target neighbour element to use in OP calculation.
        Defaults to :class:`NoneType` if no target species is known.

    neigh_idxs : :class:`list` of :class:`list` of :class:`int`
        Neighbours of each atom in site_idx. Ordering is important.
        Defaults to :class:`NoneType` for when using
        :class:`pmg.Structure` - i.e. a structure with a lattice.

    Returns:
    -------
    results : :class:`dict`
        Dictionary of format
        site_idx: dict of order parameters
        {
            `oct`: :class:`float`,
            `sq_plan`: :class:`float`,
            `q2`: :class:`float`,
            `q4`: :class:`float`,
            `q6`: :class:`float`
        }.

    """
    results = {}

    if target_species_type is None:
        targ_species = None
    else:
        targ_species = pmg.Species(target_species_type)

    # Define local order parameters class based on desired types.
    types = [
        "sq_plan",  # Square planar envs.
    ]
    loc_ops = LocalStructOrderParams(
        types=types,
    )
    if neigh_idxs is None:
        for site in site_idxs:
            site_results = loc_ops.get_order_parameters(
                structure=molecule, n=site, target_spec=[targ_species]
            )
            results[site] = {
                i: j for i, j in zip(types, site_results, strict=False)
            }
    else:
        for site, neigh in zip(site_idxs, neigh_idxs, strict=False):
            site_results = loc_ops.get_order_parameters(
                structure=molecule,
                n=site,
                indices_neighs=neigh,
                target_spec=targ_species,
            )
            results[site] = {
                i: j for i, j in zip(types, site_results, strict=False)
            }

    return results


def get_order_values(mol, metal, per_site=False):
    """Calculate order parameters around metal centres.

    Parameters
    ----------
    mol : :class:`stk.ConstructedMolecule`
        stk molecule to analyse.

    metal : :class:`int`
        Element number of metal atom.

    per_site : :class:`bool`
        Defaults to False. True if the OPs for each site are desired.

    Returns:
    -------
    results : :class:`dict`
        Dictionary of order parameter max/mins/averages if `per_site`
        is False.

    """
    pmg_mol = convert_stk_to_pymatgen(stk_mol=mol)
    # Get sites of interest and their neighbours.
    sites = []
    neighs = []
    for atom in mol.get_atoms():
        if atom.get_atomic_number() == metal:
            sites.append(atom.get_id())
            bonds = [
                i
                for i in mol.get_bonds()
                if i.get_atom1().get_id() == atom.get_id()
                or i.get_atom2().get_id() == atom.get_id()
            ]
            a_neigh = []
            for b in bonds:
                if b.get_atom1().get_id() == atom.get_id():
                    a_neigh.append(b.get_atom2().get_id())
                elif b.get_atom2().get_id() == atom.get_id():
                    a_neigh.append(b.get_atom1().get_id())
            neighs.append(a_neigh)

    order_values = calculate_sites_order_values(
        molecule=pmg_mol,
        site_idxs=sites,
        neigh_idxs=neighs,
    )

    if per_site:
        results = order_values
        return results
    else:
        # Get max, mins and averages of all OPs for the whole molecule.
        OPs = [order_values[i].keys() for i in order_values][0]
        OP_lists = {}
        for OP in OPs:
            OP_lists[OP] = [order_values[i][OP] for i in order_values]

        results = {
            # OP: (min, max, avg)
            i: {
                "min": min(OP_lists[i]),
                "max": max(OP_lists[i]),
                "avg": np.average(OP_lists[i]),
            }
            for i in OP_lists
        }

        return results


def get_xtb_energy(molecule, name, charge, calc_dir, solvent):
    if solvent is None:
        solvent_model = "alpb"
        solvent_str = None
        solvent_grid = "verytight"
        solvent_list = "gas"
        output_dir = os.path.join(calc_dir, f"{name}_xtbey")
        output_file = os.path.join(calc_dir, f"{name}_xtb.ey")
    else:
        solvent_model = "alpb"
        solvent_str = solvent
        solvent_grid = "verytight"
        solvent_list = f"{solvent_str}/{solvent_model}/{solvent_grid}"
        output_dir = os.path.join(calc_dir, f"{name}_{solvent_str}_xtbey")
        output_file = os.path.join(calc_dir, f"{name}_{solvent_str}_xtb.ey")

    if os.path.exists(output_file):
        with open(output_file) as f:
            lines = f.readlines()
        for line in lines:
            energy = float(line.rstrip())
            break
    else:
        logging.info(f"xtb energy calculation of {name} with {solvent_list}")
        xtb = stko.XTBEnergy(
            xtb_path=xtb_path(),
            output_dir=output_dir,
            gfn_version=2,
            num_cores=6,
            charge=charge,
            num_unpaired_electrons=0,
            unlimited_memory=True,
            solvent_model=solvent_model,
            solvent=solvent_str,
            solvent_grid=solvent_grid,
        )
        energy = xtb.get_energy(mol=molecule)
        with open(output_file, "w") as f:
            f.write(f"{energy}\n")

    # In a.u.
    return energy


class XTBSasa(stko.XTBEnergy):
    def _write_detailed_control(self) -> None:
        string = f"$gbsa\n   gbsagrid={self._solvent_grid}\n"
        string += "$write\n   gbsa=true"

        with open("det_control.in", "w") as f:
            f.write(string)


def get_xtb_sasa(molecule, name, charge, calc_dir, solvent):
    solvent_model = "alpb"
    solvent_str = solvent
    solvent_grid = "verytight"
    solvent_list = f"{solvent_str}/{solvent_model}/{solvent_grid}"
    output_dir = os.path.join(calc_dir, f"{name}_{solvent_str}_xtbsasa")
    output_file = os.path.join(calc_dir, f"{name}_{solvent_str}_sasa.json")

    if os.path.exists(output_file):
        with open(output_file) as f:
            sasa_data = json.load(f)

    else:
        logging.info(f"xtb sasa calculation of {name} with {solvent_list}")
        xtb = XTBSasa(
            xtb_path=xtb_path(),
            output_dir=output_dir,
            gfn_version=2,
            num_cores=6,
            charge=charge,
            num_unpaired_electrons=0,
            unlimited_memory=True,
            solvent_model=solvent_model,
            solvent=solvent_str,
            solvent_grid=solvent_grid,
        )
        next(xtb.calculate(mol=molecule))

        with open(os.path.join(output_dir, "energy.output")) as f:
            lines = f.readlines()
        switch = False
        sasa_data = []

        for line in lines:
            if switch and " total SASA /" in line:
                total_sasa = float(line.strip().split()[-1])
                break
            if switch:
                if "#" not in line:
                    sasa_data.append(line.strip().split())
            if " * generalized Born model for continuum solvation" in line:
                switch = True

        print(sasa_data)
        sasa_data = {
            int(i[0]): {
                "Z": int(i[1]),
                "element": i[2],
                "born/A": float(i[3]),
                "sasa/A2": float(i[4]),
                "hbond": float(i[5]),
            }
            for i in sasa_data
            if len(i) > 1
        }
        sasa_data["total_sasa/A2"] = total_sasa

        with open(output_file, "w") as f:
            json.dump(sasa_data, f)

    # In a.u.
    return sasa_data


def get_xtb_gsasa(name, calc_dir, solvent):
    solvent_model = "alpb"
    solvent_str = solvent
    solvent_grid = "verytight"
    solvent_list = f"{solvent_str}/{solvent_model}/{solvent_grid}"
    output_dir = os.path.join(calc_dir, f"{name}_{solvent_str}_xtbey")
    output_file = os.path.join(calc_dir, f"{name}_{solvent_str}_xtb.ey")

    if not os.path.exists(output_file):
        raise RuntimeError(
            f"xtb with {solvent} needs to have been run giving {output_dir}"
        )

    logging.info(f"xtb energy calculation of {name} with {solvent_list}")
    energy_file = os.path.join(output_dir, "energy.output")

    # Based on XX
    # Get the  nonpolar solvation contribution, which we divide by num. Pd
    # later.

    nums = re.compile(r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")
    with open(energy_file) as f:
        for line in f.readlines():
            if "-> Gsasa" in line:
                string = nums.search(line.rstrip())
                gsasa = float(string.group(0))
                break

    # In a.u.
    return gsasa


def get_xtb_gsolv(name, calc_dir, solvent):
    solvent_model = "alpb"
    solvent_str = solvent
    solvent_grid = "verytight"
    solvent_list = f"{solvent_str}/{solvent_model}/{solvent_grid}"
    output_dir = os.path.join(calc_dir, f"{name}_{solvent_str}_xtbey")
    output_file = os.path.join(calc_dir, f"{name}_{solvent_str}_xtb.ey")

    if not os.path.exists(output_file):
        raise RuntimeError(
            f"xtb with {solvent} needs to have been run giving {output_dir}"
        )

    logging.info(f"xtb energy calculation of {name} with {solvent_list}")
    energy_file = os.path.join(output_dir, "energy.output")

    nums = re.compile(r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")
    with open(energy_file) as f:
        for line in f.readlines():
            if ":: -> Gsolv" in line:
                string = nums.search(line.rstrip())

                gsolv = float(string.group(0))
                break

    # In a.u.
    return gsolv


def get_xtb_free_energy(molecule, name, charge, calc_dir, solvent):
    if solvent is None:
        solvent_model = "alpb"
        solvent_str = None
        solvent_grid = "verytight"
        solvent_list = "gas"
        output_dir = os.path.join(calc_dir, f"{name}_xtbfey")
        output_file = os.path.join(calc_dir, f"{name}_xtb.fey")
        freq_file = os.path.join(calc_dir, f"{name}_xtb.freq")
    else:
        solvent_model = "alpb"
        solvent_str = solvent
        solvent_grid = "verytight"
        solvent_list = f"{solvent_str}/{solvent_model}/{solvent_grid}"
        output_dir = os.path.join(calc_dir, f"{name}_{solvent_str}_xtbfey")
        output_file = os.path.join(calc_dir, f"{name}_{solvent_str}_xtb.fey")
        freq_file = os.path.join(calc_dir, f"{name}_{solvent_str}_xtb.freq")

    if os.path.exists(output_file):
        with open(output_file) as f:
            lines = f.readlines()
        for line in lines:
            total_free_energy = float(line.rstrip())
            break
    else:
        logging.info(
            f"xtb free energy calculation of {name} with {solvent_list}"
        )
        xtb = stko.XTBEnergy(
            xtb_path=xtb_path(),
            output_dir=output_dir,
            gfn_version=2,
            num_cores=6,
            charge=charge,
            calculate_free_energy=True,
            num_unpaired_electrons=0,
            unlimited_memory=True,
            solvent_model=solvent_model,
            solvent=solvent_str,
            solvent_grid=solvent_grid,
        )
        xtb_results = xtb.get_results(molecule)
        total_results = xtb_results.get_total_free_energy()
        total_free_energy = total_results[0]
        total_frequencies = xtb_results.get_frequencies()
        with open(output_file, "w") as f:
            f.write(f"{total_results[0]}\n")
        with open(freq_file, "w") as f:
            for freq in total_frequencies[0]:
                f.write(f"{freq}\n")

    # In a.u.
    return total_free_energy


def get_xtb_enthalpy(molecule, name, charge, calc_dir, solvent):
    solvent_model = "alpb"
    solvent_str = solvent
    solvent_grid = "verytight"
    solvent_list = f"{solvent_str}/{solvent_model}/{solvent_grid}"
    output_dir = os.path.join(calc_dir, f"{name}_{solvent_str}_xtbfey")
    output_file = os.path.join(calc_dir, f"{name}_{solvent_str}_xtb.fey")

    if not os.path.exists(output_file):
        raise RuntimeError(
            f"xtb with {solvent} needs to have been run giving {output_dir}"
        )

    logging.info(f"xtb energy calculation of {name} with {solvent_list}")
    energy_file = os.path.join(output_dir, "energy.output")

    nums = re.compile(r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")
    with open(energy_file) as f:
        for line in f.readlines():
            if "TOTAL ENTHALPY" in line:
                string = nums.search(line.rstrip())

                enthalpy = float(string.group(0))
                break

    # In a.u.
    return enthalpy


def get_dft_energy(name, txt_file):
    with open(txt_file) as f:
        lines = f.readlines()
    for line in lines:
        if name in line:
            number = line.strip().split()[1]
            if number == "None":
                return None
            else:
                energy = float(number)
                # kJmol-1
                return energy


def get_xtb_strain(
    molecule,
    name,
    liga_dir,
    calc_dir,
    exp_lig,
    solvent,
):
    ls_file = os.path.join(calc_dir, f"{name}_strain_xtb.json")
    if os.path.exists(ls_file):
        with open(ls_file) as f:
            strain_energies = json.load(f)
        return strain_energies
    strain_energies = {}

    # Define the free ligand properties.
    topo, ligand1_name, ligand2_name = name_parser(name)
    ligand1 = stk.BuildingBlock.init_from_file(
        path=str(liga_dir / f"{ligand1_name}_lowe.mol"),
    )
    ligand1_free_l_energy = get_xtb_energy(
        molecule=ligand1,
        name=f"{ligand1_name}_lowe",
        charge=0,
        calc_dir=calc_dir,
        solvent=solvent,
    )
    ligand1_smiles = stk.Smiles().get_key(ligand1)

    if ligand2_name is not None:
        ligand2 = stk.BuildingBlock.init_from_file(
            path=str(liga_dir / f"{ligand2_name}_lowe.mol"),
        )
        ligand2_free_l_energy = get_xtb_energy(
            molecule=ligand2,
            name=f"{ligand2_name}_lowe",
            charge=0,
            calc_dir=calc_dir,
            solvent=solvent,
        )
        ligand2_smiles = stk.Smiles().get_key(ligand2)
        smiles_map = {
            ligand1_smiles: ligand1_free_l_energy,
            ligand2_smiles: ligand2_free_l_energy,
        }
    else:
        smiles_map = {
            ligand1_smiles: ligand1_free_l_energy,
        }

    org_ligs, smiles_keys = get_organic_linkers(
        cage=molecule,
        metal_atom_nos=(46,),
        file_prefix=f"{name}_sg",
        calc_dir=calc_dir,
    )

    num_unique_ligands = len(set(smiles_keys.values()))
    if num_unique_ligands != exp_lig:
        raise UnexpectedNumLigands(
            f"{name} had {num_unique_ligands} unique ligands"
            f", {exp_lig} were expected. Suggests bad "
            "optimization. Recommend reoptimising structure."
        )

    for lfile in org_ligs:
        lname = lfile.replace(".mol", "")
        lmol = org_ligs[lfile]
        extracted_smiles = stk.Smiles().get_key(lmol)
        # Extracted ligand energies.
        extracted_energy = get_xtb_energy(
            lmol,
            f"{lname}_xtb",
            0,
            calc_dir,
            solvent=solvent,
        )

        # Free ligand energies.
        free_l_energy = smiles_map[extracted_smiles]

        strain = extracted_energy - free_l_energy
        strain_energies[lname] = strain

    with open(ls_file, "w") as f:
        json.dump(strain_energies, f, indent=4)
    return strain_energies


def get_pore_angle(molecule, metal_atom_num):
    atom_ids = [
        i.get_id()
        for i in molecule.get_atoms()
        if i.get_atomic_number() == metal_atom_num
    ]
    if len(atom_ids) != 2:
        raise ValueError(f"{len(atom_ids)} metal atoms found. Expecting 2")

    centroid = molecule.get_centroid()
    v1, v2 = (i - centroid for i in molecule.get_atomic_positions(atom_ids))
    aniso_angle = np.degrees(angle_between(v1, v2))

    return aniso_angle


def get_mm_distance(molecule, metal_atom_num):
    atom_ids = [
        i.get_id()
        for i in molecule.get_atoms()
        if i.get_atomic_number() == metal_atom_num
    ]
    if len(atom_ids) != 2:
        raise ValueError(f"{len(atom_ids)} metal atoms found. Expecting 2")

    position_matrix = molecule.get_position_matrix()

    distance = euclidean(
        u=position_matrix[atom_ids[0]], v=position_matrix[atom_ids[1]]
    )

    return float(distance)
