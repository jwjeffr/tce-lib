from dataclasses import dataclass
from functools import cached_property, lru_cache

import numpy as np
import sparse

from .constants import LatticeStructure, STRUCTURE_TO_ATOMIC_BASIS, STRUCTURE_TO_CUTOFF_LISTS, STRUCTURE_TO_THREE_BODY_LABELS
from . import topology


@dataclass(eq=True, frozen=True)
class Supercell:

    """
    class representing a simulation supercell. eq=True and frozen=True ensures we can hash a Supercell instance, which
    we need to cache the topology tensors later
    """

    lattice_structure: LatticeStructure
    lattice_parameter: float
    size: tuple[int, int, int]

    @cached_property
    def num_sites(self) -> int:

        """number of total lattice sites (NOT number of unit cells!)"""

        return np.prod(self.size) * STRUCTURE_TO_ATOMIC_BASIS[self.lattice_structure].shape[0]

    @cached_property
    def positions(self) -> np.typing.NDArray[np.floating]:

        """
        positions of lattice sites
        create a meshgrid of unit cell positions, and add lattice sites at atomic basis positions in each unit cell
        """

        i, j, k = (np.arange(s) for s in self.size)

        unit_cell_positions = np.array(np.meshgrid(i, j, k, indexing='ij')).reshape(3, -1).T
        positions = unit_cell_positions[:, np.newaxis, :] + \
            STRUCTURE_TO_ATOMIC_BASIS[self.lattice_structure][np.newaxis, :, :]
        return self.lattice_parameter * positions.reshape(-1, 3)

    @lru_cache
    def adjacency_tensors(self, max_order: int, tolerance: float = 1.0e-6) -> sparse.COO:

        """
        two-body adjacency tensors $A_{ij}^{(n)}$. computed by binning interatomic distances
        """

        return topology.get_adjacency_tensors(
            positions=self.positions,
            boxsize=self.lattice_parameter * np.array(self.size),
            max_distance=self.lattice_parameter * STRUCTURE_TO_CUTOFF_LISTS[self.lattice_structure][max_order],
            lattice_parameter=self.lattice_parameter,
            lattice_structure=self.lattice_structure,
            max_adjacency_order=max_order,
            tolerance=tolerance
        )

    @lru_cache
    def three_body_tensors(self, max_order: int) -> sparse.COO:

        """
        three-body tensors, computed by summing the two-body tensors

        each three-body tensor is defined by a set of labels. e.g., in an fcc solid, the first-order triplet is formed
        by three first-nearest neighbor pairs, so its label is (0, 0, 0). similarly, the second-order triplet in fcc is
        formed by two first-nearest neighbor pairs, and one second-nearest neighbor pair, so its label is (0, 0, 1). we
        sum over the different permutations (which triple-counts triplets, which is fine), and then stack them over
        the labels
        """

        three_body_labels = [
            STRUCTURE_TO_THREE_BODY_LABELS[self.lattice_structure][order] for order in range(max_order)
        ]

        return topology.get_three_body_tensors(
            lattice_structure=self.lattice_structure,
            adjacency_tensors=self.adjacency_tensors(max_order=np.concatenate(three_body_labels).max() + 1),
            max_three_body_order=max_order
        )

    def feature_vector(
        self,
        state_matrix: np.typing.NDArray[np.integer],
        max_adjacency_order: int,
        max_triplet_order: int
    ) -> np.typing.NDArray[np.integer]:

        """
        feature vector extracting topological features. fancy name for number of bonds, and number of triplets
        """

        return topology.get_feature_vector(
            adjacency_tensors=self.adjacency_tensors(max_order=max_adjacency_order),
            three_body_tensors=self.three_body_tensors(max_order=max_triplet_order),
            state_matrix=state_matrix
        )

    def clever_feature_diff(
        self,
        initial_state_matrix: np.typing.NDArray[np.integer],
        final_state_matrix: np.typing.NDArray[np.integer],
        max_adjacency_order: int,
        max_triplet_order: int,
    ) -> np.typing.NDArray[np.floating]:

        """
        clever shortcut for computing feature vector difference between two nearby states. tldr, perform a truncated
        contraction, only caring about "active" sites, or lattice sites that changed
        """

        return topology.get_feature_vector_difference(
            adjacency_tensors=self.adjacency_tensors(max_order=max_adjacency_order),
            three_body_tensors=self.three_body_tensors(max_order=max_triplet_order),
            initial_state_matrix=initial_state_matrix,
            final_state_matrix=final_state_matrix
        )
