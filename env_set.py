import pathlib


def xtb_path():
    return pathlib.Path(
        "/home/atarzia/miniconda3/envs/simple_het/bin/xtb"
    )


def project_path():
    return pathlib.Path("/home/atarzia/workingspace/simple_het/")


def liga_path():
    return project_path() / "liga"


def figu_path():
    return project_path() / "figures"


def cage_path():
    return project_path() / "cages"


def dft_path():
    return project_path() / "dft"


def calc_path():
    return project_path() / "calculations"


def crest_path():
    return project_path() / "software" / "crest"


def crest_conformer_settings(solvent=None):
    return {
        "conf_opt_level": "crude",
        "final_opt_level": "extreme",
        "charge": 0,
        "no_unpaired_e": 0,
        "max_runs": 1,
        "calc_hessian": False,
        "solvent": solvent,
        "nc": 4,
        "etemp": 300,
        "keepdir": False,
        "cross": True,
        "md_len": None,
        "ewin": 5,
        "speed_setting": None,
    }


def gulp_path():
    return pathlib.Path("/home/atarzia/software/gulp-6.1/Src/gulp")
