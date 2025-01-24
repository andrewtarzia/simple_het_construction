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
    min_s1: float
    min_s2: float
    mean_s1: float
    mean_s2: float
    len_s1: int
    len_s2: int
