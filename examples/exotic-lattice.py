from aenum import extend_enum
import numpy as np

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

    # construct the supercell
    supercell = tce.structures.Supercell(
        lattice_structure=tce.constants.LatticeStructure.DIAMOND,
        lattice_parameter=3.5,
        size=(3, 3, 3)
    )

    # construct a random state matrix, 30% A and 70% B
    rng = np.random.default_rng(seed=0)
    types = rng.choice([0, 1], p=[0.3, 0.7], size=supercell.num_sites)
    state_matrix = np.zeros((supercell.num_sites, 2))
    for site, t in enumerate(types):
        state_matrix[site, t] = 1.0

    # compute the feature vector for that state matrix
    feature_vector = supercell.feature_vector(
        state_matrix,
        max_adjacency_order=3,
        max_triplet_order=2
    )
    print(feature_vector)


if __name__ == "__main__":

    main()
