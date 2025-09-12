from ase import build
from aenum import extend_enum
import numpy as np

import tce


def main():

    fluorite_unit_cell = build.bulk("UO2", crystalstructure="fluorite", a=1.0, cubic=True)
    extend_enum(
        tce.constants.LatticeStructure,
        "FLUORITE",
        len(tce.constants.LatticeStructure) + 1
    )
    tce.constants.STRUCTURE_TO_ATOMIC_BASIS[
        tce.constants.LatticeStructure.FLUORITE
    ] = fluorite_unit_cell.positions

    supercell = fluorite_unit_cell.repeat((3, 3, 3))
    distances = np.unique(supercell.get_all_distances(mic=True).flatten())

    unique_tol = []
    for x in distances:
        if not any(np.isclose(x, u, atol=1.0e-3) for u in unique_tol):
            unique_tol.append(x)

    unique_tol = np.sort(unique_tol)
    unique_tol = unique_tol[unique_tol < 1.1]
    tce.constants.STRUCTURE_TO_CUTOFF_LISTS[
        tce.constants.LatticeStructure.FLUORITE
    ] = unique_tol
    tce.constants.STRUCTURE_TO_THREE_BODY_LABELS = tce.constants.load_three_body_labels()

    lattice_parameter = 5.6
    atoms = build.bulk(
        "UO2",
        crystalstructure="fluorite",
        a=lattice_parameter, cubic=True
    ).repeat((3, 3, 3))
    cations = np.array(["U", "Th"])
    rng = np.random.default_rng(seed=0)
    for i, symbol in enumerate(atoms.get_chemical_symbols()):
        if symbol == "O":
            continue
        atoms[i].symbol = rng.choice(cations)

    feature_vector_computer = tce.topology.topological_feature_vector_factory(
        basis=tce.constants.ClusterBasis(
            tce.constants.LatticeStructure.FLUORITE,
            lattice_parameter=lattice_parameter,
            max_adjacency_order=3,
            max_triplet_order=1
        ),
        type_map=np.append(cations, "O"),
    )
    feature_vector = feature_vector_computer(atoms)
    print(feature_vector)


if __name__ == "__main__":

    main()
