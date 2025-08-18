from typing import Callable

import pytest
import numpy as np

from tce.constants import LatticeStructure
from tce.structures import Supercell


@pytest.fixture
def get_supercell() -> Callable[[], Supercell]:

    def supercell(lattice_structure: LatticeStructure) -> Supercell:

        size = None
        if lattice_structure == LatticeStructure.SC:
            size = (5, 5, 5)
        if lattice_structure == LatticeStructure.BCC:
            size = (4, 4, 4)
        if lattice_structure == LatticeStructure.FCC:
            size = (3, 3, 3)
        if not size:
            raise ValueError("lattice_structure must be SC, BCC, or FCC")

        return Supercell(lattice_structure, lattice_parameter=1.0, size=size)

    return supercell


@pytest.mark.parametrize(
    "lattice_structure, num_expected_neighbors",
    [
        (LatticeStructure.SC, 6),
        (LatticeStructure.BCC, 8),
        (LatticeStructure.FCC, 12)
    ]
)
def test_num_neighbors(lattice_structure: LatticeStructure, num_expected_neighbors: int, get_supercell):

    supercell = get_supercell(lattice_structure)

    for adj in supercell.adjacency_tensors(max_order=1):
        assert adj.sum(axis=0).todense().mean() == num_expected_neighbors


@pytest.mark.parametrize("lattice_structure", [LatticeStructure.SC, LatticeStructure.BCC, LatticeStructure.FCC])
def test_feature_vector_shortcut(lattice_structure: LatticeStructure, get_supercell):

    rng = np.random.default_rng(seed=0)
    num_types = 3

    supercell = get_supercell(lattice_structure)

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

    feature_vector = supercell.feature_vector(
        state_matrix,
        max_adjacency_order=2,
        max_triplet_order=2
    )
    new_feature_vector = supercell.feature_vector(
        new_state_matrix,
        max_adjacency_order=2,
        max_triplet_order=2
    )
    naive_diff = new_feature_vector - feature_vector

    assert np.all(naive_diff == clever_diff)