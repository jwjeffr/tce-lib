from ase import build
import numpy as np

from tce.constants import LatticeStructure, ClusterBasis
from tce.topology import topological_feature_vector_factory


def main():

    # define the lattice
    structure = LatticeStructure.BCC
    lattice_parameter = 2.9
    size = (5, 5, 5)
    species = np.array(["Fe", "Cr"])

    # some auxiliary variables that will make things easier later
    rng = np.random.default_rng(seed=0)

    feature_vector_computer = topological_feature_vector_factory(
        basis=ClusterBasis(
            structure,
            lattice_parameter=lattice_parameter,
            max_adjacency_order=2,
            max_triplet_order=1
        ),
        type_map=species
    )

    # build the Atoms object, or alternatively load one using ase.io
    ase_supercell = build.bulk(
        species[0],
        crystalstructure=structure.name.lower(),
        a=lattice_parameter,
        cubic=True
    ).repeat(size)
    ase_supercell.symbols = rng.choice(species, p=[0.93, 0.07], size=len(ase_supercell))
    feature_vector = feature_vector_computer(ase_supercell)

    print(feature_vector)


if __name__ == "__main__":

    main()
