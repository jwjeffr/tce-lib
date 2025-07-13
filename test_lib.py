import numpy as np

from tce.constants import LatticeStructure
from tce.structures import Supercell


def main():

    rng = np.random.default_rng(seed=0)
    num_types = 3
    length = 10

    supercell = Supercell(
        lattice_structure=LatticeStructure.FCC,
        lattice_parameter=1.0,
        size=(length, length, length)
    )

    for adj in supercell.adjacency_tensors(max_order=2):
        print(adj.sum(axis=0).todense().mean())

    for thr in supercell.three_body_tensors(max_order=2):
        print(thr.sum(axis=0).sum(axis=0).todense().mean())

    types = rng.integers(num_types, size=supercell.num_sites)

    state_matrix = np.zeros((supercell.num_sites, num_types), dtype=int)
    state_matrix[np.arange(supercell.num_sites), types] = 1

    new_state_matrix = state_matrix.copy()
    first_site, second_site = rng.integers(supercell.num_sites, size=2)
    while types[first_site] == types[second_site]:
        first_site, second_site = rng.integers(supercell.num_sites, size=2)
    new_state_matrix[first_site, :] = state_matrix[second_site, :]
    new_state_matrix[second_site, :] = state_matrix[first_site, :]

    clever_diff = supercell.clever_feature_diff(
        state_matrix, new_state_matrix,
        max_adjacency_order=2, max_triplet_order=2
    )

    feature_vector = supercell.feature_vector(state_matrix, max_adjacency_order=2, max_triplet_order=2)
    new_feature_vector = supercell.feature_vector(new_state_matrix, max_adjacency_order=2, max_triplet_order=2)
    naive_diff = new_feature_vector - feature_vector

    assert np.all(naive_diff == clever_diff)


if __name__ == "__main__":

    main()
