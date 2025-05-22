from pathlib import Path
import re
from itertools import pairwise, permutations

import numpy as np
from scipy.spatial import KDTree
import sparse
from opt_einsum import contract
import matplotlib.pyplot as plt
import matplotlib as mpl

from constants import STRUCTURE_TO_THREE_BODY_LABELS, LatticeStructure


OSZICAR_PATTERN = re.compile(
    r"  ([0-9]+) F= ([+-]?(?=\.\d|\d)(?:\d+)?(?:\.?\d*))(?:[Ee]([+-]?\d+))? E0= ([+-]?(?=\.\d|\d)(?:\d+)?(?:\.?\d*))(?:[Ee]([+-]?\d+))?  d E =([+-]?(?=\.\d|\d)(?:\d+)?(?:\.?\d*))(?:[Ee]([+-]?\d+))?"
)


def main():
    root = Path(f"TaW_vasp_structs/TaW_vasp_structs/MC_structs")

    feature_vectors = []
    energies = []
    for directory in root.glob(f"*_Ta*W*"):

        energy = None
        with (directory / "OSZICAR").open("r") as file:
            for line in file:
                if not (m := OSZICAR_PATTERN.match(line)):
                    continue
                step, e, e_exponent, e0, e0_exponent, de, de_exponent = m.groups()
                energy = float(e) * 10 ** int(e_exponent)

        if not energy:
            raise ValueError
        energies.append(energy)

        poscar_path = directory / "POSCAR"
        positions = np.loadtxt(poscar_path, skiprows=8, max_rows=128)
        with (directory / "POSCAR").open("r") as file:
            lines = file.readlines()
        num_tantalum, num_tungsten = lines[6].strip().split()
        num_tantalum, num_tungsten = int(num_tantalum), int(num_tungsten)
        types = np.concatenate(
            (0 * np.ones(num_tantalum, dtype=int), 1 * np.ones(num_tungsten, dtype=int))
        )

        # compute adjacency matrix
        cell_matrix = np.loadtxt(poscar_path, skiprows=2, max_rows=3)
        positions *= 0.99
        tree = KDTree(positions, boxsize=np.diag(cell_matrix))
        distances = tree.sparse_distance_matrix(tree, max_distance=4.0).tocsr()
        distances.eliminate_zeros()
        distances = sparse.COO.from_scipy_sparse(distances)
        _, edges = np.histogram(distances.data, bins=2)

        tolerance = 0.01
        adjacency_tensors = sparse.stack([
            sparse.where(
                sparse.logical_and(distances > (1.0 - tolerance) * low, distances < (1.0 + tolerance) * high),
                x=True, y=False
            ) for low, high in pairwise(edges)
        ])

        three_body_labels = [
            STRUCTURE_TO_THREE_BODY_LABELS[LatticeStructure.BCC][order] for order in range(1)
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

        state_matrix = np.zeros((128, 2), dtype=int)
        state_matrix[np.arange(128), types] = 1
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

    predictions = feature_vectors @ cluster_interaction_vector

    inaccuracy = np.linalg.norm(predictions - energies) / np.linalg.norm(energies - energies.mean())
    pcc = 1.0 - inaccuracy ** 2
    print(pcc)
    print(len(feature_vectors))

    plt.scatter(
        energies, predictions, edgecolors="black", zorder=7, alpha=0.6
    )
    x = np.linspace(energies.min(), energies.max(), 10_000)
    plt.plot(x, x, linestyle="--", color="black", zorder=6)
    plt.grid()
    plt.xlabel("TaW DFT energy (eV)")
    plt.ylabel("TaW cluster expansion energy (eV)")
    plt.tight_layout()
    plt.savefig("fit.pdf")


if __name__ == "__main__":

    mpl.use("TkAgg")
    main()
