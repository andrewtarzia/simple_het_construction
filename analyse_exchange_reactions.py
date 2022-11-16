#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Script to analyse exchange reactions.

Author: Andrew Tarzia

"""

import logging
import sys
from itertools import product

from env_set import calc_path  # dft_path
from utilities import (
    read_xtb_energy,
    # get_dft_opt_energy,
    # get_dft_preopt_energy,
    name_parser,
)
from topologies import ligand_cage_topologies
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
        self._product_energy = self._get_energy(product_name)

    def _get_energy(self, name):
        if self._method == "xtb":
            return read_xtb_energy(
                name=name,
                calc_dir=self._calc_dir,
            )
        elif self._method == "dft_preopt":
            raise NotImplementedError()
        elif self._method == "dft":
            raise NotImplementedError()

    def _get_rxn_energies(
        self,
        lhs_stoich,
        l1_stoich,
        l1_prefix,
        l2_stoich,
        l2_prefix,
    ):
        return {
            "lhs_stoich": lhs_stoich,
            "l1": self._l1_name,
            "l2": self._l2_name,
            "l1_stoich": l1_stoich,
            "l2_stoich": l2_stoich,
            "l1_prefix": l1_prefix,
            "l2_prefix": l2_prefix,
            "lhs": lhs_stoich * self._product_energy,
            "rhs": (
                l1_stoich
                * self._get_energy(
                    name=f"{l1_prefix}_{self._l1_name}",
                )
                + l2_stoich
                * self._get_energy(
                    name=f"{l2_prefix}_{self._l2_name}",
                )
            ),
        }

    def get_all_rxn_energies(self):
        lct = ligand_cage_topologies()
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

            if l1_prefix not in lct[self._l1_name]:
                continue
            if l2_prefix not in lct[self._l2_name]:
                continue

            assert lhs_stoich * 2 == (
                l1_stoich * int(l1_prefix[1:])
                + l2_stoich * int(l2_prefix[1:])
            )
            yield self._get_rxn_energies(
                lhs_stoich=lhs_stoich,
                l1_stoich=l1_stoich,
                l1_prefix=l1_prefix,
                l2_stoich=l2_stoich,
                l2_prefix=l2_prefix,
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
    ):
        energy_p_stoich = (
            self._get_energy(
                name=f"{l_prefix}_{self._ligand_name}",
            )
            / l_stoich
        )
        return {
            "l": self._ligand_name,
            "l_stoich": l_stoich,
            "l_prefix": l_prefix,
            "energy_per_stoich": energy_p_stoich,
        }

    def get_all_rxn_energies(self):
        for l_prefix in self._pot_homo_topos:
            l_stoich = int(l_prefix[1:])

            yield self._get_rxn_energies(
                l_stoich=l_stoich,
                l_prefix=l_prefix,
            )


def main():
    if not len(sys.argv) == 1:
        logging.info(f"Usage: {__file__}\n" "   Expected 1 arguments:")
        sys.exit()
    else:
        pass

    _cd = calc_path()
    # _ld = liga_path()
    # dft_directory = dft_path()

    het_system = {
        "cis_l1_la",
        "cis_l1_lb",
        "cis_l1_lc",
        "cis_l1_ld",
        "cis_l2_ld",
        "cis_l3_ld",
    }
    lig_system = {
        "lb": ("m2", "m3", "m4"),
        "lc": ("m2", "m3", "m4"),
        "l1": ("m2", "m3", "m4", "m6"),
        "l2": ("m2", "m3", "m4", "m12"),
        "l3": ("m2", "m3", "m4", "m24", "m30"),
    }
    methods = ("xtb",)  # "dft_preopt", "dft")

    for method in methods:
        for hs in het_system:
            topo, l1, l2 = name_parser(hs)
            all_rxns = []
            erxn = ExchangeReaction(
                product_name=hs,
                reactant_names=(l1, l2),
                method=method,
                calc_dir=_cd,
            )
            for rxn in erxn.get_all_rxn_energies():
                all_rxns.append(rxn)

            plotting.plot_exchange_reactions(
                rxns=all_rxns,
                outname=f"erxns_{hs}",
            )

        for ls in lig_system:
            all_rxns = []
            erxn = HomolepticExchangeReaction(
                ligand_name=ls,
                method=method,
                calc_dir=_cd,
                pot_homo_topos=lig_system[ls],
            )
            for rxn in erxn.get_all_rxn_energies():
                all_rxns.append(rxn)

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
