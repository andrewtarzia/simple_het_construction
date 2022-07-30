#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Script to analyse exchange reactions.

Author: Andrew Tarzia

"""

import logging
import sys
import os
import json
from itertools import product
from dataclasses import dataclass

from matplotlib.colors import LightSource

from env_set import cage_path, calc_path, dft_path, liga_path
from utilities import (
    read_xtb_energy,
    get_dft_opt_energy,
    get_dft_preopt_energy,
    name_parser,
)
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
        if self._method == 'xtb':
            return read_xtb_energy(
                name=name,
                calc_dir=self._calc_dir,
            )
        elif self._method == 'dft_preopt':
            raise NotImplementedError()
        elif self._method == 'dft':
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
            'lhs_stoich': lhs_stoich,
            'l1': self._l1_name,
            'l2': self._l2_name,
            'l1_stoich': l1_stoich,
            'l2_stoich': l2_stoich,
            'l1_prefix': l1_prefix,
            'l2_prefix': l2_prefix,
            'lhs': lhs_stoich*self._product_energy,
            'rhs': (
                l1_stoich*self._get_energy(
                    name=f'{l1_prefix}_{self._l1_name}',
                )
                +
                l2_stoich*self._get_energy(
                    name=f'{l2_prefix}_{self._l2_name}',
                )
            )
        }

    def get_all_rxn_energies(self):
        pot_homo_topos = ('m2', 'm3', 'm4')
        for l1_prefix, l2_prefix in product(pot_homo_topos, repeat=2):
            if l1_prefix == l2_prefix:
                l1_stoich = 1
                l2_stoich = 1
                lhs_stoich = int(l2_prefix[1])
            else:
                l1_stoich = int(l2_prefix[1])
                l2_stoich = int(l1_prefix[1])
                lhs_stoich = int((l1_stoich * l2_stoich))

            assert lhs_stoich*2 == (
                l1_stoich*int(l1_prefix[1])
                + l1_stoich*int(l1_prefix[1])
            )
            yield self._get_rxn_energies(
                lhs_stoich=lhs_stoich,
                l1_stoich=l1_stoich,
                l1_prefix=l1_prefix,
                l2_stoich=l2_stoich,
                l2_prefix=l2_prefix,
            )


def main():
    if (not len(sys.argv) == 1):
        logging.info(
            f'Usage: {__file__}\n'
            '   Expected 1 arguments:'
        )
        sys.exit()
    else:
        pass

    _wd = cage_path()
    _cd = calc_path()
    _ld = liga_path()
    dft_directory = dft_path()

    het_system = {
        'cis_l1_la',
        'cis_l1_lb',
        'cis_l1_lc',
        'cis_l1_ld',
        'cis_l2_ld',
        'cis_l3_ld',
    }
    methods = ('xtb', 'dft_preopt', 'dft')

    for method in methods:
        print(method)
        for hs in het_system:
            print(hs)
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
                hs=hs,
                outname=f'erxns_{hs}',
            )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
    )
    main()
