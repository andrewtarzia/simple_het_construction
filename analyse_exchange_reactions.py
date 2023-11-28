#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Script to analyse exchange reactions.

Author: Andrew Tarzia

"""

import logging
import sys
import json
import os
from itertools import product

from env_set import calc_path, cage_path
from utilities import name_parser
from topologies import erxn_cage_topologies
import plotting


class ExchangeReaction:
    """
    Define methods to analyse reaction of form zA <-> xB + yC.

    """

    def __init__(self, product_name, reactant_names, method, calc_dir):
        self._product_name = product_name
        self._l1_name, self._l2_name = reactant_names
        self._calc_dir = calc_dir
        self._method = method

    def _get_energy(self, name, structure_results):
        energy = structure_results[name][self._method]

        if energy is None:
            logging.info("remmove this when you have the data")
            return None
            raise ValueError(f"{name} has no {self._method} energy!")

        if self._method in (
            "xtb_solv_opt_gasenergy_au",
            "xtb_solv_opt_dmsoenergy_au",
        ):
            # To kJmol-1
            energy *= 2625.5
        return energy

    def _get_rxn_energies(
        self,
        lhs_stoich,
        l1_stoich,
        l1_prefix,
        l2_stoich,
        l2_prefix,
        structure_results,
    ):
        try:
            lhs_energy = lhs_stoich * self._get_energy(
                name=f"{self._product_name}",
                structure_results=structure_results,
            )
            rhs_energy1 = l1_stoich * self._get_energy(
                name=f"{l1_prefix}_{self._l1_name}",
                structure_results=structure_results,
            )
            rhs_energy2 = l2_stoich * self._get_energy(
                name=f"{l2_prefix}_{self._l2_name}",
                structure_results=structure_results,
            )
        except TypeError:
            lhs_energy = 0
            rhs_energy1 = rhs_energy2 = 0
        return {
            "lhs_stoich": lhs_stoich,
            "l1": self._l1_name,
            "l2": self._l2_name,
            "l1_stoich": l1_stoich,
            "l2_stoich": l2_stoich,
            "l1_prefix": l1_prefix,
            "l2_prefix": l2_prefix,
            "lhs": lhs_energy,
            "rhs": (rhs_energy1 + rhs_energy2),
        }

    def get_all_rxn_energies(self, structure_results):
        lct = erxn_cage_topologies()
        pot_homo_topos = ("m2", "m3", "m4", "m6", "m12", "m24", "m30")
        for l1_prefix, l2_prefix in product(pot_homo_topos, repeat=2):
            if l1_prefix == l2_prefix:
                l1_stoich = 1
                l2_stoich = 1
                lhs_stoich = int(l2_prefix[1:])
            else:
                l1_stoich = int(l2_prefix[1:])
                l2_stoich = int(l1_prefix[1:])
                lhs_stoich = int((l1_stoich * l2_stoich))

            if l1_prefix != lct[self._l1_name]:
                continue
            if l2_prefix != lct[self._l2_name]:
                continue

            assert lhs_stoich * 2 == (
                l1_stoich * int(l1_prefix[1:]) + l2_stoich * int(l2_prefix[1:])
            )
            yield self._get_rxn_energies(
                lhs_stoich=lhs_stoich,
                l1_stoich=l1_stoich,
                l1_prefix=l1_prefix,
                l2_stoich=l2_stoich,
                l2_prefix=l2_prefix,
                structure_results=structure_results,
            )


class HomolepticExchangeReaction(ExchangeReaction):
    """
    Define methods to analyse reaction of form zA <-> xB.

    """

    def __init__(self, ligand_name, pot_homo_topos, method, calc_dir):
        self._ligand_name = ligand_name
        self._pot_homo_topos = pot_homo_topos
        self._calc_dir = calc_dir
        self._method = method

    def _get_rxn_energies(
        self,
        l_stoich,
        l_prefix,
        structure_results,
    ):
        energy_p_stoich = (
            self._get_energy(
                name=f"{l_prefix}_{self._ligand_name}",
                structure_results=structure_results,
            )
            / l_stoich
        )
        return {
            "l": self._ligand_name,
            "l_stoich": l_stoich,
            "l_prefix": l_prefix,
            "energy_per_stoich": energy_p_stoich,
        }

    def get_all_rxn_energies(self, structure_results):
        for l_prefix in self._pot_homo_topos:
            l_stoich = int(l_prefix[1:])

            yield self._get_rxn_energies(
                l_stoich=l_stoich,
                l_prefix=l_prefix,
                structure_results=structure_results,
            )


def main():
    if not len(sys.argv) == 1:
        logging.info(f"Usage: {__file__}\n" "   Expected 1 arguments:")
        sys.exit()
    else:
        pass

    _cd = calc_path()
    _wd = cage_path()

    structure_res_file = os.path.join(_wd, "all_structure_res.json")
    if os.path.exists(structure_res_file):
        with open(structure_res_file, "r") as f:
            structure_results = json.load(f)

    het_system = (
        "cis_l1_la",
        "cis_l1_lb",
        "cis_l1_lc",
        "cis_l1_ld",
        "cis_l2_ld",
        "cis_l3_ld",
        # "cis_ll1_ls",
        # "cis_ll2_ls",
    )
    lig_system = {
        # "lb": ("m2", "m3", "m4"),
        # "lc": ("m2", "m3", "m4"),
        "l1": ("m2", "m3", "m4", "m6", "m12", "m24"),
        # "l2": ("m12",),
        # "l3": ("m2", "m3", "m4", "m24", "m30"),
    }
    methods = (
        "xtb_solv_opt_gasenergy_au",
        "xtb_solv_opt_dmsoenergy_au",
        "pbe0_def2svp_sp_gas_kjmol",
        "pbe0_def2svp_sp_dmso_kjmol",
        "pbe0_def2svp_opt_gas_kjmol",
        "pbe0_def2svp_opt_dmso_kjmol",
    )

    all_het_exchanges = {}
    for hs in het_system:
        topo, l1, l2 = name_parser(hs)
        all_rxns = {}
        for method in methods:
            all_rxns[method] = []
            erxn = ExchangeReaction(
                product_name=hs,
                reactant_names=(l1, l2),
                method=method,
                calc_dir=_cd,
            )
            for rxn in erxn.get_all_rxn_energies(structure_results):
                all_rxns[method].append(rxn)

        plotting.plot_exchange_reactions(
            rxns=all_rxns,
            outname=f"erxns_{hs}",
        )
        all_het_exchanges[hs] = all_rxns

    plotting.plot_all_exchange_reactions(
        all_rxns=all_het_exchanges,
        outname="erxns_all",
    )

    plotting.plot_all_exchange_reactions_production(
        all_rxns=all_het_exchanges,
        outname="erxns_mainpaper",
    )

    for ls in lig_system:
        all_rxns = {}
        for method in methods:
            all_rxns[method] = []
            erxn = HomolepticExchangeReaction(
                ligand_name=ls,
                method=method,
                calc_dir=_cd,
                pot_homo_topos=lig_system[ls],
            )
            for rxn in erxn.get_all_rxn_energies(structure_results):
                all_rxns[method].append(rxn)

        plotting.plot_homoleptic_exchange_reactions(
            rxns=all_rxns,
            outname=f"erxns_{ls}",
        )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    main()
