from typing import Callable
import re

import pytest
import numpy as np
from ase import build
from ase.calculators.singlepoint import SinglePointCalculator

import tce
from tce.constants import LatticeStructure
from tce.structures import Supercell
from tce.training import (
    LimitingRidge,
    ClusterBasis,
    INCOMPATIBLE_GEOMETRY_MESSAGE,
    NO_POTENTIAL_ENERGY_MESSAGE,
    NON_CUBIC_CELL_MESSAGE,
    LARGE_SYSTEM_MESSAGE
)


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


def test_noncubic_cell_raises_value_error():

    configurations = [
        build.bulk("Fe", crystalstructure="bcc", a=2.7, cubic=False).repeat((2, 2, 2)),
        build.bulk("Cr", crystalstructure="bcc", a=2.7, cubic=False).repeat((3, 3, 3))
    ]
    for configuration in configurations:
        configuration.calc = SinglePointCalculator(configuration, energy=-1.0)

    with pytest.raises(ValueError, match=NON_CUBIC_CELL_MESSAGE):
        _ = tce.training.train(
            configurations=configurations,
            basis=ClusterBasis(
                lattice_structure=LatticeStructure.BCC,
                lattice_parameter=2.7,
                max_adjacency_order=3,
                max_triplet_order=1
            )
        )


def test_inconsistent_geometry_raises_value_error():

    configurations = [
        build.bulk("Fe", crystalstructure="bcc", a=2.7, cubic=True).repeat((2, 2, 2)),
        build.bulk("Fe", crystalstructure="fcc", a=2.7, cubic=True).repeat((3, 3, 3))
    ]

    for configuration in configurations:
        configuration.calc = SinglePointCalculator(configuration, energy=-1.0)

    with pytest.raises(ValueError, match=INCOMPATIBLE_GEOMETRY_MESSAGE):
        _ = tce.training.train(
            configurations=configurations,
            basis=ClusterBasis(
                lattice_structure=LatticeStructure.BCC,
                lattice_parameter=2.7,
                max_adjacency_order=3,
                max_triplet_order=1
            )
        )


def test_no_energy_computation_raises_value_error():

    configurations = [
        build.bulk("Fe", crystalstructure="bcc", a=2.7, cubic=True).repeat((2, 2, 2)),
        build.bulk("Cr", crystalstructure="bcc", a=2.7, cubic=True).repeat((3, 3, 3))
    ]

    with pytest.raises(ValueError, match=NO_POTENTIAL_ENERGY_MESSAGE):
        _ = tce.training.train(
            configurations=configurations,
            basis=ClusterBasis(
                lattice_structure=LatticeStructure.BCC,
                lattice_parameter=2.7,
                max_adjacency_order=3,
                max_triplet_order=1
            )
        )


def test_large_system_in_training(monkeypatch):

    with monkeypatch.context() as m:

        m.setattr("tce.training.LARGE_SYSTEM_THRESHOLD", 10)

        configurations = [
            build.bulk("Fe", crystalstructure="bcc", a=2.7, cubic=True).repeat((2, 2, 2)),
        ]
        configurations[0].calc = SinglePointCalculator(configurations[0], energy=-1.0)

        with pytest.warns(UserWarning, match=re.escape(LARGE_SYSTEM_MESSAGE)):
            _ = tce.training.train(
                configurations=configurations,
                basis=ClusterBasis(
                    lattice_structure=LatticeStructure.BCC,
                    lattice_parameter=2.7,
                    max_adjacency_order=3,
                    max_triplet_order=1
                )
            )