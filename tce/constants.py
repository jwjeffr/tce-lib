from enum import Enum, auto
from typing import Dict
from functools import cache
from itertools import product, permutations

import numpy as np
from scipy.spatial import KDTree
import sparse


class LatticeStructure(Enum):

    r"""
    Enum type defining the most typical lattice structures: simple cubic, body-centered cubic, and face-centered cubic.

    [<img
        src="https://wisc.pb.unizin.org/app/uploads/sites/293/2019/07/CNX_Chem_10_06_CubUntCll.png"
        width=100%
        alt="SC, BCC, and FCC lattice structures"
        title="Lattice structures. Source: UW-Madison Chemistry 103/104 Resource Book"
    />](https://wisc.pb.unizin.org/app/uploads/sites/293/2019/07/CNX_Chem_10_06_CubUntCll.png)

    Chiefly, this data type defines mappings between lattice structure and three body labels. This is additionally
    useful for creating a `Supercell` instance, e.g.:

    ```py
    from tce.structures.supercell import Supercell

    supercell = Supercell(
        lattice_structure=LatticeStructure.BCC,
        lattice_parameter=2.5,
        size=(4, 4, 4)
    )
    ```

    which generates a $4\times 4\times 4$ bcc supercell with lattice parameter $a = 2.5$, typically in units of
    $\mathring{\mathrm{A}}$.
    """

    SC = auto()
    r"""simple cubic lattice structure"""
    BCC = auto()
    r"""body-centered cubic lattice structure"""
    FCC = auto()
    r"""face-centered cubic lattice structure"""


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
r"""Mapping from lattice structure to atomic basis, i.e. positions of atoms within a unit cell. Here, we use the
conventional unit cell"""

STRUCTURE_TO_CUTOFF_LISTS: Dict[LatticeStructure, np.typing.NDArray[np.floating]] = {
    LatticeStructure.SC: np.array([1.0, np.sqrt(2.0), np.sqrt(3.0), 2.0]),
    LatticeStructure.BCC: np.array([0.5 * np.sqrt(3.0), 1.0, np.sqrt(2.0), 0.5 * np.sqrt(11.0)]),
    LatticeStructure.FCC: np.array([0.5 * np.sqrt(2.0), 1.0, np.sqrt(1.5), np.sqrt(2.0)])
}
r"""Mapping from lattice structure to neighbor cutoffs, in units of the lattice parameter $a$"""


def load_three_body_labels(
        tolerance: float = 0.01,
        min_num_sites: int = 125,
) -> Dict[LatticeStructure, np.typing.NDArray[np.integer]]:

    r"""
    Function to generate three body labels for a given lattice structure. Here, we compute the set of three body labels
    for all lattice structures up to fourth-nearest neighbors, and store them in a mapping allowing for
    $\mathcal{O}(1)$ access. This function is called at import, with the result stored in the module-level constant
    `STRUCTURE_TO_THREE_BODY_LABELS`.
    """

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
r"""Mapping from lattice structure to set of three body labels"""