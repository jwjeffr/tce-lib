from enum import Enum, auto
from typing import Dict
from functools import cache
from itertools import product, permutations

import numpy as np
from scipy.spatial import KDTree
import sparse


class LatticeStructure(Enum):

    """
    lattice structure enum. will be useful for defining things like cutoff distances and atomic bases
    """

    SC = auto()
    BCC = auto()
    FCC = auto()


STRUCTURE_TO_ATOMIC_BASIS: Dict[LatticeStructure, np.typing.NDArray[np.floating]] = {
    LatticeStructure.SC: np.array([
        [0.0, 0.0, 0.0]
    ]),
    LatticeStructure.BCC: np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.5]
    ]),
    LatticeStructure.FCC: np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.5, 0.5, 0.0]
    ])
}

STRUCTURE_TO_CUTOFF_LISTS: Dict[LatticeStructure, np.typing.NDArray[np.floating]] = {
    LatticeStructure.SC: np.array([1.0, np.sqrt(2.0), np.sqrt(3.0), 2.0]),
    LatticeStructure.BCC: np.array([0.5 * np.sqrt(3.0), 1.0, np.sqrt(2.0), 0.5 * np.sqrt(11.0)]),
    LatticeStructure.FCC: np.array([0.5 * np.sqrt(2.0), 1.0, np.sqrt(1.5), np.sqrt(2.0)])
}


@cache
def load_three_body_labels(
        tolerance: float = 0.01,
        min_num_sites: int = 125,
) -> Dict[LatticeStructure, np.typing.NDArray[np.integer]]:

    label_dict = {}
    for lattice_structure in LatticeStructure:

        min_num_unit_cells = min_num_sites // len(STRUCTURE_TO_ATOMIC_BASIS[lattice_structure])
        s = np.ceil(np.cbrt(min_num_unit_cells))
        size = (s, s, s)
        i, j, k = (np.arange(s) for s in size)
        unit_cell_positions = np.array(np.meshgrid(i, j, k, indexing='ij')).reshape(3, -1).T

        cutoffs = STRUCTURE_TO_CUTOFF_LISTS[lattice_structure]
        positions = unit_cell_positions[:, np.newaxis, :] + \
            STRUCTURE_TO_ATOMIC_BASIS[lattice_structure][np.newaxis, :, :]
        positions = positions.reshape(-1, 3)

        tree = KDTree(positions, boxsize=np.array(size))
        distances = tree.sparse_distance_matrix(tree, max_distance=(1.0 + tolerance) * cutoffs[-1]).tocsr()
        distances.eliminate_zeros()
        distances = sparse.COO.from_scipy_sparse(distances)

        adjacency_tensors = sparse.stack([
            sparse.where(
                sparse.logical_and(distances > (1.0 - tolerance) * c, distances < (1.0 + tolerance) * c),
                x=True, y=False
            ) for c in cutoffs
        ])

        max_adj_order = adjacency_tensors.shape[0]
        non_zero_labels = []
        for labels in product(*[range(max_adj_order) for _ in range(3)]):
            if not labels[0] <= labels[1] <= labels[2]:
                continue
            three_body_tensor = sum(
                sparse.einsum(
                    "ij,jk,ki->ijk",
                    adjacency_tensors[i],
                    adjacency_tensors[j],
                    adjacency_tensors[k]
                ) for i, j, k in set(permutations(labels))
            )
            if not three_body_tensor.nnz:
                continue
            non_zero_labels.append(list(labels))

        non_zero_labels.sort(key=lambda x: (max(x), x))
        label_dict[lattice_structure] = np.array(non_zero_labels)

    return label_dict


STRUCTURE_TO_THREE_BODY_LABELS = load_three_body_labels()
