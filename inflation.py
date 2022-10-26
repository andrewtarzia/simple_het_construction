#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Module for pore mapper functions.

Author: Andrew Tarzia

"""

import logging
import pore_mapper as pm


class PoreMapper:
    def __init__(self, name, calc_dir):
        self._name = name
        self._calc_dir = calc_dir

    def get_results(self, molecule):
        results = {}

        xyz_file = str(self._calc_dir / f"{self._name}_pm.xyz")
        final_stru_xyz = str(
            self._calc_dir / f"{self._name}_pm_stru.xyz"
        )
        final_blob_xyz = str(
            self._calc_dir / f"{self._name}_pm_blob.xyz"
        )
        final_pore_xyz = str(
            self._calc_dir / f"{self._name}_pm_pore.xyz"
        )

        logging.info(f"running pore mapper on {self._name}:")

        # Read in host from xyz file.
        molecule.write(xyz_file)
        host = pm.Host.init_from_xyz_file(path=xyz_file)
        host = host.with_centroid([0.0, 0.0, 0.0])

        # Define calculator object.
        calculator = pm.Inflater(bead_sigma=1.2)

        # Run calculator on host object, analysing output.
        final_result = calculator.get_inflated_blob(host=host)
        pore = final_result.pore
        blob = final_result.pore.get_blob()
        windows = pore.get_windows()

        results["num_windows"] = len(windows)
        results["max_window_size"] = max(windows)
        results["min_window_size"] = min(windows)
        results["pore_volume"] = pore.get_volume()
        results["asphericity"] = pore.get_asphericity()
        results[
            "shape_anisotropy"
        ] = pore.get_relative_shape_anisotropy()

        # Do final structure.
        host.write_xyz_file(final_stru_xyz)
        blob.write_xyz_file(final_blob_xyz)
        pore.write_xyz_file(final_pore_xyz)

        return results
