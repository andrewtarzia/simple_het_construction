"""Code to extract a topology from a molecule."""

import pathlib

import stk
import stko
from definitions import Study1EnvVariables
from rdkit.Chem import AllChem as rdkit  # noqa: N813


def extract_topo(
    struct: stk.Molecule,
    smarts: str,
    prefix: str,
    working_dir: pathlib.Path,
) -> stko.TopologyInfo:
    """Extract a topology."""
    struct = struct.with_centroid([0, 0, 0])

    broken_bonds_by_id = []
    disconnectors = []
    rdkit_mol = struct.to_rdkit_mol()
    rdkit.SanitizeMol(rdkit_mol)
    for atom_ids in rdkit_mol.GetSubstructMatches(
        query=rdkit.MolFromSmarts(smarts),
    ):
        bond_1 = atom_ids[0]
        bond_2 = atom_ids[1]
        broken_bonds_by_id.append(sorted((bond_1, bond_2)))
        disconnectors.extend((bond_1, bond_2))

    new_topology_graph = stko.TopologyExtractor()
    tg_info = new_topology_graph.extract_topology(
        molecule=struct,
        broken_bonds_by_id=broken_bonds_by_id,
        disconnectors=set(disconnectors),
    )
    struct.write(str(working_dir / f"{prefix}_tg.mol"))
    tg_info.write(str(working_dir / f"{prefix}_tg.pdb"))
    return tg_info


def main() -> None:
    """Run script."""
    prefix = "1482268"
    _wd = Study1EnvVariables.xtal_path

    bonds = []
    with (_wd / f"{prefix}_SandPdonly.bonds").open("r") as f:
        for line in f:
            bline = line.strip()
            ids = bline.split()[1:]
            new_bline = (int(ids[0]) + 1, int(ids[1]) + 1)
            bonds.append(new_bline)

    cage = stk.BuildingBlock.init_from_file(
        str(_wd / f"{prefix}_SandPdonly.pdb")
    )
    atom_tuple = tuple(cage.get_atoms())
    cage = stk.BuildingBlock.init(
        atoms=atom_tuple,
        bonds=tuple(
            stk.Bond(
                atom1=atom_tuple[aid1 - 1],
                atom2=atom_tuple[aid2 - 1],
                order=1,
            )
            for aid1, aid2 in bonds
        ),
        position_matrix=cage.get_position_matrix(),
    )

    cage.write(str(_wd / f"{prefix}_SandPdonlybonded.mol"))
    topo_info = extract_topo(
        struct=cage,
        prefix=prefix,
        smarts="[Pd]~[S]",
        working_dir=_wd,
    )

    with (_wd / f"{prefix}_tginfo.txt").open("w") as f:
        edges = topo_info.get_edge_pairs()
        edge_string = ""
        count_bonds = {}
        for i, pair in enumerate(edges):
            if pair[0] not in count_bonds:
                count_bonds[pair[0]] = 0
            if pair[1] not in count_bonds:
                count_bonds[pair[1]] = 0
            count_bonds[pair[0]] += 1
            count_bonds[pair[1]] += 1
            edge_string += (
                "stk.Edge(\n"
                f"    id={i},\n"
                f"    vertex1=_vertex_prototypes[{pair[0]}],\n"
                f"    vertex2=_vertex_prototypes[{pair[1]}],\n"
                "),\n"
            )

        vertex_string = ""
        two_fg_ids = []
        four_fg_ids = []
        v_pos = topo_info.get_vertex_positions()
        for i in v_pos:
            pos = v_pos[i]
            bond_count = count_bonds[i]
            if bond_count == 2:  # noqa: PLR2004
                vtype = "stk.cage.AngledVertex"
                two_fg_ids.append(i)
            elif bond_count == 4:  # noqa: PLR2004
                vtype = "stk.cage.NonLinearVertex"
                four_fg_ids.append(i)
            vertex_string += (
                f"{vtype}(\n"
                f"    id={i},\n"
                f"    position={[round(i, 1) for i in pos]},\n"
                "),\n"
            )

        bbdict = (
            f"metal: {tuple(four_fg_ids)},\n" f"ligand: {tuple(two_fg_ids)},\n"
        )

        f.write("bbdict:\n")
        f.write(bbdict)
        f.write("==============================\n")
        f.write(vertex_string)
        f.write(edge_string)


if __name__ == "__main__":
    main()
