"""Define environment variables."""

import pathlib

res_str = "$r$"
mean_res_str = r"$r_{\mathrm{avg}}$"
mean_res1_str = r"$r_{1,\mathrm{avg}}$"
mean_res2_str = r"$r_{2,\mathrm{avg}}$"

experimental_ligand_outcomes = {
    ("e16_0", "e10_0"): {"success": True, "g": [], "1-r": []},
    ("e16_0", "e17_0"): {"success": True, "g": [], "1-r": []},
    ("e10_0", "e17_0"): {"success": False, "g": [], "1-r": []},
    ("e11_0", "e10_0"): {"success": True, "g": [], "1-r": []},
    ("e16_0", "e14_0"): {"success": True, "g": [], "1-r": []},
    ("e18_0", "e14_0"): {"success": True, "g": [], "1-r": []},
    ("e18_0", "e10_0"): {"success": True, "g": [], "1-r": []},
    ("e12_0", "e10_0"): {"success": True, "g": [], "1-r": []},
    ("e11_0", "e14_0"): {"success": True, "g": [], "1-r": []},
    ("e12_0", "e14_0"): {"success": True, "g": [], "1-r": []},
    ("e11_0", "e13_0"): {"success": True, "g": [], "1-r": []},
    ("e12_0", "e13_0"): {"success": True, "g": [], "1-r": []},
    ("e13_0", "e14_0"): {"success": False, "g": [], "1-r": []},
    ("e11_0", "e12_0"): {"success": False, "g": [], "1-r": []},
    ("sla_0", "sl1_0"): {"success": False, "g": [], "1-r": []},
    ("slb_0", "sl1_0"): {"success": True, "g": [], "1-r": []},
    ("slc_0", "sl1_0"): {"success": True, "g": [], "1-r": []},
    ("sld_0", "sl1_0"): {"success": False, "g": [], "1-r": []},
    ("sla_0", "sl2_0"): {"success": False, "g": [], "1-r": []},
    ("slb_0", "sl2_0"): {"success": False, "g": [], "1-r": []},
    ("slc_0", "sl2_0"): {"success": False, "g": [], "1-r": []},
    ("sld_0", "sl2_0"): {"success": False, "g": [], "1-r": []},
    ("sla_0", "sl3_0"): {"success": False, "g": [], "1-r": []},
    ("slb_0", "sl3_0"): {"success": False, "g": [], "1-r": []},
    ("slc_0", "sl3_0"): {"success": False, "g": [], "1-r": []},
    ("sld_0", "sl3_0"): {"success": False, "g": [], "1-r": []},
    # Chand.
    ("lab_0", "la_0"): {"success": True, "g": [], "1-r": []},
    ("lab_0", "lb_0"): {"success": False, "g": [], "1-r": []},
    ("lab_0", "lc_0"): {"success": False, "g": [], "1-r": []},
    ("lab_0", "ld_0"): {"success": True, "g": [], "1-r": []},
    # Molinska.
    ("m2h_0", "m4q_0"): {"success": True, "g": [], "1-r": []},
    ("m2h_0", "m4p_0"): {"success": True, "g": [], "1-r": []},
}


class MatchingSettings:
    """Define the variables for matching calculations."""

    # Matching settings.
    # method = "L-BFGS-B"  # noqa: ERA001
    # bounds = ((-5, 10), (-10, 10), (-45, 45)) # noqa: ERA001
    vector_length = 2.02
    method = None
    bounds = None
    set_state = [-5, 0, 0]  # noqa: RUF012
    initial_guesses = ([0, 0, 20], [0, 0, -20], [0, 0, 0])


class EnvVariables:
    """Define the variables for working environment."""

    dihedral_cutoff = 10
    strain_cutoff = 5
    study1_dihedral_cutoff = 10
    rmsd_threshold = 0.2

    k_bond = 1
    k_angle = 5

    gbeta = 10
    rbeta = 1
    found_max_r_works = 13.242133537811274

    xtb_path = pathlib.Path("/home/atarzia/miniforge3/envs/cpl/bin/xtb")
    gulp_path = pathlib.Path("/home/atarzia/software/gulp-6.1.2/Src/gulp")


class Study1EnvVariables:
    """Define the variables for working environment."""

    dihedral_cutoff = 10
    strain_cutoff = 5
    rmsd_threshold = 0.2

    project_path = pathlib.Path("/home/atarzia/workingspace/simple_het/")
    liga_path = project_path / "liga"
    figu_path = project_path / "figures"
    cage_path = project_path / "cages"
    xtal_path = project_path / "xtals"
    calc_path = project_path / "calculations"
    gulp_path = pathlib.Path("/home/atarzia/software/gulp-6.1/Src/gulp")
    xtb_path = pathlib.Path("/home/atarzia/miniconda3/envs/simple_het/bin/xtb")
