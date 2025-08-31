from dataclasses import dataclass
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, TypeAlias, Union, Optional
import warnings

import numpy as np
from scipy.spatial import KDTree
from ase import Atoms

from tce.constants import LatticeStructure, STRUCTURE_TO_CUTOFF_LISTS, STRUCTURE_TO_ATOMIC_BASIS
from tce.topology import get_adjacency_tensors, get_three_body_tensors, get_feature_vector


NON_CUBIC_CELL_MESSAGE = "At least one of your configurations has a non-cubic cell. For now, tce-lib does not support non-cubic lattices."

INCOMPATIBLE_GEOMETRY_MESSAGE = "Geometry in all configurations must match geometry in cluster basis."

NO_POTENTIAL_ENERGY_MESSAGE = "At least one of your configurations does not have a computable potential energy."

LARGE_SYSTEM_THRESHOLD = 1_000
LARGE_SYSTEM_MESSAGE = f"You have passed a relatively large system (larger than {LARGE_SYSTEM_THRESHOLD:.0f}) as a training point. This will be very slow."


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


PropertyComputer: TypeAlias = Callable[[Atoms], Union[float, np.typing.NDArray[np.floating]]]

def total_energy(atoms: Atoms) -> float:

    try:
        return atoms.get_potential_energy()
    except RuntimeError as e:
        raise ValueError(NO_POTENTIAL_ENERGY_MESSAGE) from e


def get_data_pairs(
    configurations: list[Atoms],
    basis: ClusterBasis,
    target_property_computer: Optional[PropertyComputer] = None
) -> tuple[np.typing.NDArray[np.floating], np.typing.NDArray[np.floating]]:

    basis_atomic_volume = basis.lattice_parameter ** 3 / len(STRUCTURE_TO_ATOMIC_BASIS[basis.lattice_structure])
    for configuration in configurations:

        if np.any(configuration.get_cell().angles() != 90.0 * np.ones(3)):
            raise ValueError(NON_CUBIC_CELL_MESSAGE)

        configuration_atomic_volume = configuration.get_volume() / len(configuration)
        if not np.isclose(configuration_atomic_volume, basis_atomic_volume):
            raise ValueError(INCOMPATIBLE_GEOMETRY_MESSAGE)

        if len(configuration) > LARGE_SYSTEM_THRESHOLD:
            warnings.warn(LARGE_SYSTEM_MESSAGE, UserWarning)

    if not target_property_computer:
        target_property_computer = total_energy

    # not all configurations need to have the same number of types, calculate the union of types
    all_types = set.union(*(set(x.get_chemical_symbols()) for x in configurations))
    type_map = np.array(sorted(list(all_types)))

    num_types = len(type_map)
    inverse_type_map = {symbol: i for i, symbol in enumerate(type_map)}

    feature_size = basis.max_adjacency_order * num_types ** 2 + basis.max_triplet_order * num_types ** 3
    X = np.zeros((len(configurations), feature_size))
    y: list[Union[float, np.typing.NDArray[np.floating]]] = [np.nan] * len(configurations)

    for index, atoms in enumerate(configurations):

        y[index] = target_property_computer(atoms)

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

    return X, np.array(y)


@dataclass
class TrainingMethod(ABC):

    r"""
    Abstract base class for defining how to train a model $y = \beta^\intercal X$. $X$ here is **not** a state matrix,
    but rather a data matrix.
    """

    @abstractmethod
    def fit(
        self,
        configurations: list[Atoms],
        basis: ClusterBasis,
        property_computer: Optional[PropertyComputer] = None
    ) -> CEModel:

        r"""
        Train method for model training.

        Args:
            configurations (list[Atoms]):
                list of configurations to train the model for
            basis (ClusterBasis):
                trained model basis
            property_computer (PropertyComputer):
                optional property computer, which computes a property from an `ase.Atoms` object. if not specified,
                set to compute total energy
        """

        pass


class LimitingRidge(TrainingMethod):

    r"""
    Train by minimizing the limiting ridge problem:

    $$ L(\beta\;|\;\lambda) = \|X\beta - y \|_2^2 + \lambda \|\beta\|_2^2 $$

    $$ \hat{\beta} = \lim_{\lambda\to 0^+}\arg\min_{\beta} L(\beta\;|\;\lambda) = X^+ y $$

    where $X^+$ denotes the [Moore-Penrose inverse](https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse).
    """

    def fit(
        self,
        configurations: list[Atoms],
        basis: ClusterBasis,
        property_computer: Optional[PropertyComputer] = None
    ) -> CEModel:

        r"""
        Train method for model training.

        Args:
            configurations (list[Atoms]):
                list of configurations to train the model for
            basis (ClusterBasis):
                trained model basis
            property_computer (PropertyComputer):
                optional property computer, which computes a property from an `ase.Atoms` object. if not specified,
                set to compute total energy
        """

        # not all configurations need to have the same number of types, calculate the union of types
        all_types = set.union(*(set(x.get_chemical_symbols()) for x in configurations))
        type_map = np.array(sorted(list(all_types)))

        X, y = get_data_pairs(configurations, basis, target_property_computer=property_computer)

        return CEModel(cluster_basis=basis, interaction_vector=np.linalg.pinv(X) @ y, type_map=type_map)
