import numpy as np
import matplotlib.pyplot as plt

import tce


def main():
    vertical_lengths = np.arange(3, 9)
    num_atoms = np.zeros_like(vertical_lengths)
    feature_sizes = np.zeros_like(vertical_lengths)
    rng = np.random.default_rng(seed=0)

    for i, length in enumerate(vertical_lengths):
        # construct the supercell
        supercell = tce.structures.Supercell(
            lattice_structure=tce.constants.LatticeStructure.FCC,
            lattice_parameter=3.5,
            size=(3, 3, length)
        )
        num_atoms[i] = supercell.num_sites

        types = rng.choice([0, 1], p=[0.5, 0.5], size=supercell.num_sites)
        state_matrix = np.zeros((supercell.num_sites, 2))
        for site, t in enumerate(types):
            state_matrix[site, t] = 1.0

        # compute the feature vector for that state matrix
        feature_vector = supercell.feature_vector(
            state_matrix,
            max_adjacency_order=3,
            max_triplet_order=2
        )
        feature_sizes[i] = np.linalg.norm(feature_vector)

    plt.scatter(num_atoms, feature_sizes, edgecolor="black", facecolor="turquoise", zorder=7)
    plt.xlabel("Number of atoms")
    plt.ylabel(r"Feature magnitude $\|\mathbf{t}\|$")
    plt.grid()
    plt.tight_layout()
    plt.savefig("size-dependence.png", dpi=800, bbox_inches="tight")


if __name__ == "__main__":

    main()
