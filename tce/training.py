from dataclasses import dataclass
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from scipy.spatial import KDTree
from ase import Atoms

from tce.constants import LatticeStructure, STRUCTURE_TO_CUTOFF_LISTS
from tce.topology import get_adjacency_tensors, get_three_body_tensors, get_feature_vector


@dataclass
class ClusterBasis:

    lattice_structure: LatticeStructure
    r"""lattice structure that the trained model corresponds to"""

    lattice_parameter: float
    r"""lattice parameter that the trained model corresponds to"""

    max_adjacency_order: int
    r"""maximum adjacency order (number of nearest neighbors) that the trained model accounts for"""

    max_triplet_order: int
    r"""maximum triplet order (number of three-body clusters) that the trained model accounts for"""


@dataclass
class CEModel:

    cluster_basis: ClusterBasis
    r"""Cluster basis for the trained model"""

    interaction_vector: np.typing.NDArray[np.floating]
    r"""trained interaction vector"""

    type_map: np.typing.NDArray[np.str_]
    r"""array of chemical species, e.g. `np.array(["Fe", "Cr"])`"""

    def save(self, path: Path) -> None:

        np.savez(
            path,
            interaction_vector=self.interaction_vector,
            type_map=self.type_map,
            lattice_structure=self.cluster_basis.lattice_structure.name.lower(),
            lattice_parameter=self.cluster_basis.lattice_parameter,
            max_adjacency_order=self.cluster_basis.max_adjacency_order,
            max_triplet_order=self.cluster_basis.max_triplet_order
        )

    @classmethod
    def load(cls, path: Path) -> "CEModel":

        data = np.load(path)
        basis = ClusterBasis(
            lattice_structure=getattr(LatticeStructure, data["lattice_structure"].item().upper()),
            lattice_parameter=data["lattice_parameter"].item(),
            max_adjacency_order=data["max_adjacency_order"].item(),
            max_triplet_order=data["max_triplet_order"].item()
        )
        return cls(
            cluster_basis=basis,
            interaction_vector=data["interaction_vector"],
            type_map=data["type_map"],
        )


def get_data_pairs(
    configurations: list[Atoms],
    basis: ClusterBasis
) -> tuple[np.typing.NDArray[np.floating], np.typing.NDArray[np.floating]]:

    # not all configurations need to have the same number of types, calculate the union of types
    all_types = set.union(*(set(x.get_chemical_symbols()) for x in configurations))
    type_map = np.array(sorted(list(all_types)))

    num_types = len(type_map)
    inverse_type_map = {symbol: i for i, symbol in enumerate(type_map)}

    feature_size = basis.max_adjacency_order * num_types ** 2 + basis.max_triplet_order * num_types ** 3
    X = np.zeros((len(configurations), feature_size))
    y = np.zeros(len(configurations))

    for index, atoms in enumerate(configurations):
        tree = KDTree(atoms.positions, boxsize=np.diag(atoms.cell))
        adjacency_tensors = get_adjacency_tensors(
            tree=tree,
            cutoffs=basis.lattice_parameter * STRUCTURE_TO_CUTOFF_LISTS[basis.lattice_structure][
                                              :basis.max_adjacency_order],
        )
        three_body_tensors = get_three_body_tensors(
            lattice_structure=basis.lattice_structure,
            adjacency_tensors=adjacency_tensors,
            max_three_body_order=basis.max_triplet_order,
        )

        state_matrix = np.zeros((len(atoms), num_types))
        for site, symbol in enumerate(atoms.symbols):
            state_matrix[site, inverse_type_map[symbol]] = 1.0

        # compute the feature vector and store it
        X[index, :] = get_feature_vector(
            adjacency_tensors=adjacency_tensors,
            three_body_tensors=three_body_tensors,
            state_matrix=state_matrix
        )

        y[index] = atoms.get_potential_energy()

    return X, y


@dataclass
class TrainingMethod(ABC):

    r"""
    Abstract base class for defining how to train a model $y = \beta^\intercal X$. $X$ here is **not** a state matrix,
    but rather a data matrix.
    """

    @abstractmethod
    def fit(self, configurations: list[Atoms], basis: ClusterBasis) -> CEModel:

        r"""
        Train method for model training.

        Args:
            configurations (list[Atoms]):
                list of configurations to train the model for
            basis (ClusterBasis):
                trained model basis
        """

        pass


class LimitingRidge(TrainingMethod):

    r"""
    Train by minimizing the limiting ridge problem:

    $$ L(\beta\;|\;\lambda) = \|X\beta - y \|_2^2 + \lambda \|\beta\|_2^2 $$

    $$ \hat{\beta} = \lim_{\lambda\to 0^+}\arg\min_{\beta} L(\beta\;|\;\lambda) = X^+ y $$

    where $X^+$ denotes the [Moore-Penrose inverse](https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse).
    """

    def fit(self, configurations: list[Atoms], basis: ClusterBasis) -> CEModel:

        r"""
        Train method for model training.

        Args:
            configurations (list[Atoms]):
                list of configurations to train the model for
            basis (ClusterBasis):
                trained model basis
        """

        # not all configurations need to have the same number of types, calculate the union of types
        all_types = set.union(*(set(x.get_chemical_symbols()) for x in configurations))
        type_map = np.array(sorted(list(all_types)))

        X, y = get_data_pairs(configurations, basis)

        return CEModel(cluster_basis=basis, interaction_vector=np.linalg.pinv(X) @ y, type_map=type_map)
