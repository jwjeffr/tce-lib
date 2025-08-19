from ase import build
import numpy as np
from scipy.spatial import KDTree

from tce.constants import LatticeStructure, STRUCTURE_TO_CUTOFF_LISTS
from tce.topology import get_adjacency_tensors, get_three_body_tensors, get_feature_vector


def main():

    # define the lattice
    structure = LatticeStructure.BCC
    lattice_parameter = 2.9
    concentration = {"Fe": 0.93, "Cr": 0.07}
    size = (5, 5, 5)

    # some auxiliary variables that will make things easier later
    species = list(concentration.keys())
    concentrations = list(concentration.values())
    inverse_type_map = {symbol: i for i, symbol in enumerate(species)}
    rng = np.random.default_rng(seed=0)

    # build the Atoms object, or alternatively load one using ase.io
    ase_supercell = build.bulk(
        species[0],
        crystalstructure=structure.name.lower(),
        a=lattice_parameter,
        cubic=True
    ).repeat(size)
    ase_supercell.symbols = rng.choice(species, p=concentrations, size=len(ase_supercell))

    # tce-lib needs a KDTree object, so create one and use this to compute topological tensors
    tree = KDTree(ase_supercell.positions, boxsize=np.diag(ase_supercell.cell))
    adjacency_tensors = get_adjacency_tensors(
        tree=tree,
        cutoffs=lattice_parameter * STRUCTURE_TO_CUTOFF_LISTS[structure][:2]
    )
    # compute the number of n'th-nearest neighbors as a sanity check. should be 8 first-nearest neighbors and 6
    # second-nearest neighbors
    for adj in adjacency_tensors:
        print(adj.sum(axis=0).mean())
    three_body_tensors = get_three_body_tensors(
        lattice_structure=structure,
        adjacency_tensors=adjacency_tensors,
        max_three_body_order=1
    )

    # define a state matrix X from the object, i.e. a one-hot encoding of the species list
    state_matrix = np.zeros((len(ase_supercell), len(species)))
    for site, symbol in enumerate(ase_supercell.symbols):
        state_matrix[site, inverse_type_map[symbol]] = 1.0

    # compute the feature vector and print it
    feature_vector = get_feature_vector(
        adjacency_tensors=adjacency_tensors,
        three_body_tensors=three_body_tensors,
        state_matrix=state_matrix
    )
    print(feature_vector)


if __name__ == "__main__":

    main()
