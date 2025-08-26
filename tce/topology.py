from itertools import permutations
from typing import Sequence, Optional

from scipy.spatial import KDTree
import numpy as np
import sparse
from opt_einsum import contract

from .constants import LatticeStructure, STRUCTURE_TO_THREE_BODY_LABELS, load_three_body_labels


def symmetrize(tensor: sparse.COO, axes: Optional[tuple[int, ...]] =None) -> sparse.COO:
    r"""
    symmetrize a tensor $T$:

    $$T_{(i_1 i_2 \cdots i_r)} = \frac{1}{r!}\sum_{\sigma\in S_n} T_{\sigma(i_1) \sigma(i_2) \cdots \sigma(i_r)}$$

    Where $S_n$ is the symmetric group on $n$ elements, so we are summing over the permutations of the indices.

    E.g. $T_{(12)} = \frac{T_{12} + T_{21}}{2}$, or equivalently $\text{symmetrize}(T) = \frac{T + T^\intercal}{2}$

    Specify the `axes` argument if you only want to symmetrize over a subset of indices

    Args:
        tensor (sparse.COO):
            The tensor $T$ to symmetrize
        axes (tuple[int]):
            The axes over which to symmetrize. If not provided, symmetrize over all axes. Defaults to `None`.
    """

    if not axes:
        axes = tuple(range(tensor.ndim))

    perms = list(permutations(axes))

    return sum(sparse.moveaxis(tensor, axes, perm) for perm in perms) / len(perms)


def get_adjacency_tensors(
        tree: KDTree,
        cutoffs: Sequence[float],
        tolerance: float = 0.01
) -> sparse.COO:

    r"""
    compute adjacency tensors $A_{ij}^{(n)}$. we first compute the sparse distance matrix using the
    `scipy.spatial.KDTree` data structure, and then convert to a `sparse.COO` tensor. then, we stack according to
    neighbor order, i.e. $A_{ij}^{(n)} = 1$ if sites $i$ and $j$ are $n$'th order neighbors, and $0$ else.

    Args:
        tree (scipy.spatial.KDTree):
            The KDTree to compute adjacency tensors from. This structure stores lattice positions as well as lattice
            vectors to encode periodic boundary conditions.
        cutoffs (Sequence[float]):
            Distance cutoffs for interatomic distances.
        tolerance (float):
            The tolerance $\varepsilon$ to include when binning interatomic distances. For example, when searching
            for a neighbor at distance $d$, we search in the shell $[(1 - \varepsilon)d, (1 + \varepsilon)d]$. This
            should be a small number. Defaults to $0.01$.
    """

    distances = tree.sparse_distance_matrix(tree, max_distance=(1.0 + tolerance) * cutoffs[-1]).tocsr()
    distances.eliminate_zeros()
    distances_sp = sparse.COO.from_scipy_sparse(distances)

    return sparse.stack([
        sparse.where(
            sparse.logical_and(distances_sp > (1.0 - tolerance) * c, distances_sp < (1.0 + tolerance) * c),
            x=True, y=False
        ) for c in cutoffs
    ])


def get_three_body_tensors(
        lattice_structure: LatticeStructure,
        adjacency_tensors: sparse.COO,
        max_three_body_order: int,
) -> sparse.COO:

    r"""
    compute three-body tensors $B_{ijk}^{(n)}$:

    $$ B_{ijk}^{(n)} = \bigvee_{\sigma\in S_3}
        A_{ij}^{(\sigma(\mathfrak{a}))}A_{jk}^{(\sigma(\mathfrak{b}))}A_{ki}^{(\sigma(\mathfrak{c}))} $$

    where the mapping between $n$ and $(\mathfrak{a}, \mathfrak{b}, \mathfrak{c})$ is defined by
    `constants.STRUCTURE_TO_THREE_BODY_LABELS`.

    Args:
        lattice_structure (LatticeStructure):
            The lattice structure to compute three-body tensors from. This argument is chiefly used here to grab three
            body labels, which depend on lattice structure.
        adjacency_tensors (sparse.COO):
            Adjacency tensors $A_{ij}^{(n)}$ of shape `(number of neighbors, number of sites, number of sites)`
        max_three_body_order (int):
            Maximum neighbor order of three-body tensors.
    """

    try:
        labels = STRUCTURE_TO_THREE_BODY_LABELS[lattice_structure]
    except KeyError:
        labels = load_three_body_labels()[lattice_structure]

    three_body_labels = [
        labels[order] for order in range(max_three_body_order)
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

    return three_body_tensors


def get_feature_vector(
        adjacency_tensors: sparse.COO,
        three_body_tensors: sparse.COO,
        state_matrix: np.typing.NDArray
) -> np.typing.NDArray:

    r"""
    topological feature vector $\mathbf{t}$ with components $N_{\alpha\beta}^{(n)} = A_{ij}^{(n)}X_{i\alpha}X_{j\beta}$
    and $M_{\alpha\beta\gamma}^{(n)} = B_{ijk}^{(n)} X_{i\alpha}X_{j\beta}X_{k\gamma}$.

    Args:
        adjacency_tensors (sparse.COO):
            Adjacency tensors $A_{ij}^{(n)}$ of shape `(number of neighbors, number of sites, number of sites)`.
        three_body_tensors (sparse.COO):
            Three body tensors $B_{ijk}^{(n)}$ of shape
            `(number of neighbors, number of sites, number of sites, number of sites, number of sites)`.
        state_matrix (np.ndarray):
            The state tensor $\mathbf{X}$, defined by $X_{i\alpha} = [\text{site $i$ occupied by type $\alpha$}]$,
            where $[\cdot]$ is the [Iverson bracket](https://en.wikipedia.org/wiki/Iverson_bracket).
    """

    return np.concatenate([
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


def get_feature_vector_difference(
        adjacency_tensors: sparse.COO,
        three_body_tensors: sparse.COO,
        initial_state_matrix: np.typing.NDArray,
        final_state_matrix: np.typing.NDArray
) -> np.typing.NDArray:

    r"""
    shortcut method for computing feature vector difference
    $\Delta\mathbf{t} = \mathbf{t}(\mathbf{X}') - \mathbf{t}(\mathbf{X})$ between two nearby states

    Args:
        adjacency_tensors (sparse.COO):
            Adjacency tensors $A_{ij}^{(n)}$ of shape `(number of neighbors, number of sites, number of sites)`.
        three_body_tensors (sparse.COO):
            Three body tensors $B_{ijk}^{(n)}$ of shape
            `(number of neighbors, number of sites, number of sites, number of sites, number of sites)`.
        initial_state_matrix (np.typing.NDArray):
            The initial state tensor $\mathbf{X}$.
        final_state_matrix (np.typing.NDArray):
            The final state tensor $\mathbf{X}'$.
    """

    sites, _ = np.where(initial_state_matrix != final_state_matrix)
    sites = np.unique(sites).tolist()

    truncated_adj = sparse.take(adjacency_tensors, sites, axis=1)
    initial_feature_vec_truncated = 2 * symmetrize(contract(
        "nij,iα,jβ->nαβ",
        truncated_adj,
        initial_state_matrix[sites, :],
        initial_state_matrix
    ), axes=(1, 2)).flatten()
    final_feature_vec_truncated = 2 * symmetrize(contract(
        "nij,iα,jβ->nαβ",
        truncated_adj,
        final_state_matrix[sites, :],
        final_state_matrix
    ), axes=(1, 2)).flatten()

    if three_body_tensors is None:
        return final_feature_vec_truncated - initial_feature_vec_truncated

    truncated_thr = sparse.take(three_body_tensors, sites, axis=1)
    initial_feature_vec_truncated = np.concatenate(
        [
            initial_feature_vec_truncated,
            3 * symmetrize(contract(
                "nijk,iα,jβ,kγ->nαβγ",
                truncated_thr,
                initial_state_matrix[sites, :],
                initial_state_matrix,
                initial_state_matrix
            ), axes=(1, 2, 3)).flatten()
        ]
    )
    final_feature_vec_truncated = np.concatenate(
        [
            final_feature_vec_truncated,
            3 * symmetrize(contract(
                "nijk,iα,jβ,kγ->nαβγ",
                truncated_thr,
                final_state_matrix[sites, :],
                final_state_matrix,
                final_state_matrix
            ), axes=(1, 2, 3)).flatten()
        ]
    )
    return final_feature_vec_truncated - initial_feature_vec_truncated
