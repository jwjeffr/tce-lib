from pathlib import Path
from itertools import permutations

import numpy as np
from scipy.spatial import KDTree
import sparse
from opt_einsum import contract

from constants import STRUCTURE_TO_THREE_BODY_LABELS, LatticeStructure, STRUCTURE_TO_CUTOFF_LISTS


def main():
    root = Path("CoNiCrFeMn_lammps")

    feature_vectors = []
    energies = []
    for directory in root.glob("samples/*"):

        energy = np.loadtxt(directory / "energy.txt")
        energies.append(energy)

        atomic_data_path = directory / "unrelaxed.dat"
        atomic_data = np.loadtxt(atomic_data_path, skiprows=19, max_rows=300)
        types = atomic_data[:, 1].astype(int)
        positions = atomic_data[:, 2:5].astype(float)

        boxsize = [0, 0, 0]
        with atomic_data_path.open("r") as file:
            for line in file:
                if "xlo xhi" in line:
                    xlo, xhi, _, __ = line.split()
                    boxsize[0] = float(xhi)
                    continue
                if "ylo yhi" in line:
                    ylo, yhi, _, __ = line.split()
                    boxsize[1] = float(yhi)
                    continue
                if "zlo zhi" in line:
                    zlo, zhi, _, __ = line.split()
                    boxsize[2] = float(zhi)
                    continue
        tree = KDTree(positions, boxsize=boxsize)
        distances = tree.sparse_distance_matrix(tree, max_distance=4.0).tocsr()
        distances.eliminate_zeros()
        distances = sparse.COO.from_scipy_sparse(distances)

        cutoffs = [3.59 * c for c in STRUCTURE_TO_CUTOFF_LISTS[LatticeStructure.FCC]]
        cutoffs.pop(0)

        tolerance = 0.01
        adjacency_tensors = sparse.stack([
            sparse.where(
                sparse.logical_and(distances > (1.0 - tolerance) * c, distances < (1.0 + tolerance) * c),
                x=True, y=False
            ) for c in cutoffs[:2]
        ])

        three_body_labels = [
            STRUCTURE_TO_THREE_BODY_LABELS[LatticeStructure.FCC][order] for order in range(1)
        ]

        three_body_tensors = sparse.stack([
            sum(
                sparse.einsum(
                    "ij,jk,ki->ijk",
                    adjacency_tensors[i],
                    adjacency_tensors[j],
                    adjacency_tensors[k]
                ) for i, j, k in set(permutations(labels))
            ) for labels in three_body_labels
        ])

        state_matrix = np.zeros((300, 5), dtype=int)
        state_matrix[np.arange(300), types - 1] = 1
        feature_vector = np.concatenate([
            contract(
                "nij,iα,jβ->nαβ",
                adjacency_tensors,
                state_matrix,
                state_matrix
            ).flatten(),
            contract(
                "nijk,iα,jβ,kγ->nαβγ",
                three_body_tensors,
                state_matrix,
                state_matrix,
                state_matrix
            ).flatten()
        ])

        feature_vectors.append(feature_vector)

    feature_vectors = np.array(feature_vectors)
    energies = np.array(energies)
    
    cluster_interaction_vector = np.linalg.pinv(feature_vectors) @ energies
    inaccuracy = np.linalg.norm(energies - feature_vectors @ cluster_interaction_vector)
    inaccuracy /= np.linalg.norm(energies - energies.mean())
    pcc = 1.0 - inaccuracy ** 2
    
    print(pcc)
    np.savetxt("CoNiCrFeMn-interaction-vector.txt", cluster_interaction_vector, fmt="%.18f")


if __name__ == "__main__":

    main()

