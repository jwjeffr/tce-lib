from dataclasses import dataclass
from functools import cached_property, lru_cache
from time import perf_counter_ns

import numpy as np
import sparse
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import linregress

from tce.constants import LatticeStructure, STRUCTURE_TO_ATOMIC_BASIS, STRUCTURE_TO_CUTOFF_LISTS, STRUCTURE_TO_THREE_BODY_LABELS
from tce import topology


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
        return positions.reshape(-1, 3)

    @lru_cache
    def adjacency_tensors(self, max_order: int, tolerance: float = 1.0e-6) -> sparse.COO:

        """
        two-body adjacency tensors $A_{ij}^{(n)}$. computed by binning interatomic distances
        """

        return topology.get_adjacency_tensors_shelling(
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


def main():

    rng = np.random.default_rng(seed=0)
    num_types = 3

    lengths = np.arange(3, 15)

    # insert a dummy length to cache einsum expressions
    lengths = np.insert(lengths, 0, lengths.min())
    naive_timings = np.zeros_like(lengths)
    clever_timings = np.zeros_like(lengths)
    num_atoms = np.zeros_like(lengths)

    for i, length in enumerate(lengths):

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

        past = perf_counter_ns()
        clever_diff = supercell.clever_feature_diff(
            state_matrix, new_state_matrix,
            max_adjacency_order=2, max_triplet_order=2
        )
        clever_timings[i] = perf_counter_ns() - past

        past = perf_counter_ns()
        feature_vector = supercell.feature_vector(state_matrix, max_adjacency_order=2, max_triplet_order=2)
        new_feature_vector = supercell.feature_vector(new_state_matrix, max_adjacency_order=2, max_triplet_order=2)
        naive_diff = new_feature_vector - feature_vector
        naive_timings[i] = perf_counter_ns() - past

        num_atoms[i] = supercell.num_sites

        assert np.all(naive_diff == clever_diff)

    # first one is a dummy call
    clever_timings = clever_timings[1:] / 1.0e+9
    naive_timings = naive_timings[1:] / 1.0e+9
    num_atoms = num_atoms[1:]

    clever_reg = linregress(np.log(num_atoms), np.log(clever_timings))
    naive_reg = linregress(np.log(num_atoms), np.log(naive_timings))

    plt.scatter(
        num_atoms, clever_timings, edgecolors="black", facecolor="mediumpurple", zorder=6,
        label=rf"clever, $\mathcal{{O}}\left(N^{{{clever_reg.slope:.2f} \pm {clever_reg.stderr:.2f}}}\right)$"
    )
    plt.scatter(
        num_atoms, naive_timings, edgecolors="black", facecolor="seagreen", zorder=6,
        label=rf"naive, $\mathcal{{O}}\left(N^{{{naive_reg.slope:.2f} \pm {naive_reg.stderr:.2f}}}\right)$"
    )

    num_atoms_continuous = np.linspace(num_atoms.min(), num_atoms.max(), 1_000)
    predicted_clever_times = np.exp(clever_reg.intercept) * num_atoms_continuous ** clever_reg.slope
    plt.plot(num_atoms_continuous, predicted_clever_times, linestyle="--", color="mediumpurple")

    predicted_naive_times = np.exp(naive_reg.intercept) * num_atoms_continuous ** naive_reg.slope
    plt.plot(num_atoms_continuous, predicted_naive_times, linestyle="--", color="seagreen")

    plt.yscale("log")
    plt.xscale("log")
    plt.ylabel(r"compute time of $\Delta \mathbf{t}$ (seconds)")
    plt.xlabel(r"number of atoms $N$")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("perf.pdf", bbox_inches="tight")


if __name__ == "__main__":

    mpl.use("TkAgg")
    main()
