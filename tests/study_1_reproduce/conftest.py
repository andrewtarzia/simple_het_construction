import pytest

from .case_data import CaseData

ligand_smiles = {
    # Diverging.
    "l1": "C1=NC=CC(C2=CC=C3OC4C=CC(C5C=CN=CC=5)=CC=4C3=C2)=C1",
    "l2": "C1=CC(=CC(=C1)C2=CC=NC=C2)C3=CC=NC=C3",
    # Converging.
    "lb": (
        "C1=CN=CC2C(C3=CC=C(C#CC4N=C(C#CC5=CC=C(C6=CC=CC7C=CN=CC6="
        "7)C=C5)C=CC=4)C=C3)=CC=CC1=2"
    ),
    "lc": (
        "C1C2=C(C(=CC=C2)C2C=CC(C#CC3=CC=CC(C#CC4C=CC(C5C6=C(C=CN="
        "C6)C=CC=5)=CC=4)=C3)=CC=2)C=NC=1"
    ),
    "ld": (
        "C1C2=C(C(=CC=C2)C2C=CC(C#CC3=CC=C(C#CC4C=CC(C5C6=C(C=CN=C"
        "6)C=CC=5)=CC=4)S3)=CC=2)C=NC=1"
    ),
}


@pytest.fixture(
    scope="session",
    params=(
        lambda name: CaseData(
            l1_name="l1",
            l2_name="lb",
            l1_smiles=ligand_smiles["l1"],
            l2_smiles=ligand_smiles["lb"],
            low_energies={
                "l1": ("1", 417.67521599764564),
                "lb": ("462", 395.52410740095047),
            },
            min_gs={"l1,lb": 0.08},
            mean_g=0.35809283135486325,
            std_g=0.14427758135380753,
            name=name,
        ),
        lambda name: CaseData(
            l1_name="l1",
            l2_name="lc",
            l1_smiles=ligand_smiles["l1"],
            l2_smiles=ligand_smiles["lc"],
            low_energies={
                "l1": ("1", 417.67521599764564),
                "lc": ("272", 398.24277072062586),
            },
            min_gs={"l1,lc": 0.05},
            mean_g=0.34547840288270687,
            std_g=0.1486257940592086,
            name=name,
        ),
        lambda name: CaseData(
            l1_name="l1",
            l2_name="ld",
            l1_smiles=ligand_smiles["l1"],
            l2_smiles=ligand_smiles["ld"],
            low_energies={
                "l1": ("1", 417.67521599764564),
                "ld": ("208", 495.5184877635414),
            },
            min_gs={"l1,ld": 0.12},
            mean_g=0.3164837915978819,
            std_g=0.09811604512153681,
            name=name,
        ),
        lambda name: CaseData(
            l1_name="l2",
            l2_name="lb",
            l1_smiles=ligand_smiles["l2"],
            l2_smiles=ligand_smiles["lb"],
            low_energies={
                "l2": ("19", 189.08845518079497),
                "lb": ("462", 395.52410740095047),
            },
            min_gs={"l2,lb": 0.09},
            mean_g=0.7375756889206226,
            std_g=0.22352121626371566,
            name=name,
        ),
        lambda name: CaseData(
            l1_name="l2",
            l2_name="lc",
            l1_smiles=ligand_smiles["l2"],
            l2_smiles=ligand_smiles["lc"],
            low_energies={
                "l2": ("19", 189.08845518079497),
                "lc": ("272", 398.24277072062586),
            },
            min_gs={"l2,lc": 0.11},
            mean_g=0.7024982945170168,
            std_g=0.25907715343348386,
            name=name,
        ),
        lambda name: CaseData(
            l1_name="l2",
            l2_name="ld",
            l1_smiles=ligand_smiles["l2"],
            l2_smiles=ligand_smiles["ld"],
            low_energies={
                "l2": ("19", 189.08845518079497),
                "ld": ("208", 495.5184877635414),
            },
            min_gs={"l2,ld": 0.5},
            mean_g=0.735503721280968,
            std_g=0.11340096377664843,
            name=name,
        ),
    ),
)
def case_data(request: pytest.FixtureRequest) -> CaseData:
    return request.param(
        f"{request.fixturename}{request.param_index}",
    )
