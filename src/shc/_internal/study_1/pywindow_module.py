"""Module for pywindow functions."""

import json
import pathlib

import pywindow as pw  # pyright: ignore  # noqa: PGH003
import stk


class PyWindow:
    def __init__(self, name: str, calc_dir: pathlib.Path) -> None:
        self._name = name
        self._calc_dir = calc_dir

    def get_results(self, molecule: stk.Molecule) -> dict:
        results = {}

        xyz_file = str(self._calc_dir / f"{self._name}_pw.xyz")
        json_file = str(self._calc_dir / f"{self._name}_pw.json")
        pdb_file = str(self._calc_dir / f"{self._name}_pw.pdb")

        # Read in host from xyz file.
        molecule.write(xyz_file)
        if json_file.exists():
            with json_file.open("r") as f:
                results = json.load(f)
        else:
            molsys = pw.MolecularSystem.load_file(xyz_file)
            mol = molsys.system_to_molecule()
            try:
                mol.calculate_pore_diameter_opt()
                mol.calculate_pore_volume_opt()
                mol.calculate_windows()
                mol.calculate_centre_of_mass()
                results = {
                    "pore_diameter_opt": (
                        mol.properties["pore_diameter_opt"]["diameter"]
                    ),
                    "pore_volume_opt": (mol.properties["pore_volume_opt"]),
                    "windows": tuple(
                        i for i in mol.properties["windows"]["diameters"]
                    ),
                }
                mol.dump_molecule(
                    pdb_file,
                    include_coms=True,
                    override=True,
                )
            except ValueError:
                results = {
                    "pore_diameter_opt": 0,
                    "pore_volume_opt": 0,
                    "windows": (),
                }
                mol.dump_molecule(
                    pdb_file,
                    include_coms=False,
                    override=True,
                )

            with json_file.open("w") as f:
                json.dump(results, f)

        return results
