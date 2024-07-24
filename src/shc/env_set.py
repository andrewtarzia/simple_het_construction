import pathlib


def xtb_path():
    return pathlib.Path("/home/atarzia/miniconda3/envs/simple_het/bin/xtb")


def project_path():
    return pathlib.Path("/home/atarzia/workingspace/simple_het/")


def liga_path():
    return project_path() / "liga"


def figu_path():
    return project_path() / "figures"


def cage_path():
    return project_path() / "cages"


def xtal_path():
    return project_path() / "xtals"


def calc_path():
    return project_path() / "calculations"


def gulp_path():
    return pathlib.Path("/home/atarzia/software/gulp-6.1/Src/gulp")
