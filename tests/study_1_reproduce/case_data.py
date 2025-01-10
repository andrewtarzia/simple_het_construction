from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CaseData:
    l1_name: str
    l2_name: str
    l1_smiles: str
    l2_smiles: str
    low_energies: dict[str, tuple[str, float]]
    min_gs: dict[str, float]
    mean_g: float
    std_g: float
    name: str
