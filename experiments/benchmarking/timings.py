from time import perf_counter_ns

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import linregress

from tce.constants import LatticeStructure
from tce.structures import Supercell


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
