from aenum import extend_enum
import numpy as np
from ase import build

import tce


def main():

    # extend the LatticeStructure Enum class to include DIAMOND
    extend_enum(tce.constants.LatticeStructure, "DIAMOND", len(tce.constants.LatticeStructure) + 1)

    # define the mapping from structure to atomic basis
    tce.constants.STRUCTURE_TO_ATOMIC_BASIS[tce.constants.LatticeStructure.DIAMOND] = np.array([
        [0.0, 0.0, 0.0],
        [0.25, 0.25, 0.25],
        [0.0, 0.5, 0.5],
        [0.25, 0.75, 0.75],
        [0.5, 0.0, 0.5],
        [0.75, 0.25, 0.75],
        [0.5, 0.5, 0.0],
        [0.75, 0.75, 0.25]
    ])

    # define the cutoff list for the new structure
    # you can compute these distances using ASE if you do not know them a-priori
    tce.constants.STRUCTURE_TO_CUTOFF_LISTS[tce.constants.LatticeStructure.DIAMOND] = np.array([
        0.25 * np.sqrt(3.0), 0.5 * np.sqrt(2.0), 0.25 * np.sqrt(11.0), 1.0
    ])

    # then, reload the three body labels
    tce.constants.STRUCTURE_TO_THREE_BODY_LABELS = tce.constants.load_three_body_labels()

    species = np.array(["Si", "Ge"])
    lattice_parameter = 5.5

    feature_vector_computer = tce.topology.topological_feature_vector_factory(
        basis=tce.constants.ClusterBasis(
            tce.constants.LatticeStructure.DIAMOND,
            lattice_parameter=lattice_parameter,
            max_adjacency_order=2,
            max_triplet_order=1
        ),
        type_map=species
    )

    rng = np.random.default_rng(seed=0)
    atoms = build.bulk(
        species[0],
        crystalstructure="diamond",
        a=lattice_parameter,
        cubic=True
    ).repeat((3, 3, 3))
    atoms.symbols = rng.choice(species, p=[0.3, 0.7], size=len(atoms))

    feature_vector = feature_vector_computer(atoms)
    print(feature_vector)

    # check the number of nearest neighbors
    # for cubic diamond, we should see 4th 1st nearest, 12 2nd nearest, 12 3rd nearest, and 6th 4th nearest
    # we should also see that there's no dispersity in the nearest neighbor counts
    distances = atoms.get_all_distances(mic=True)
    cutoffs = lattice_parameter * tce.constants.STRUCTURE_TO_CUTOFF_LISTS[tce.constants.LatticeStructure.DIAMOND]
    tol = 1.0e-3
    for i, cutoff in enumerate(cutoffs):
        num_neighbors = np.logical_and(
            (1.0 - tol) * cutoff < distances, distances < (1.0 + tol) * cutoff
        ).sum(axis=0)
        print(num_neighbors.mean(), num_neighbors.std())


if __name__ == "__main__":

    main()
