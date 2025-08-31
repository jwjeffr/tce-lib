from pathlib import Path

from ase import build, Atoms
from ase.calculators.eam import EAM
import numpy as np

from tce.constants import LatticeStructure
from tce.training import ClusterBasis, LimitingRidge
from tce.structures import Supercell


def compute_stresses(atoms: Atoms) -> np.typing.NDArray[np.floating]:

    # train on "extensive" stress - feature vectors are extensive

    try:
        return len(atoms) * atoms.get_stress()
    except RuntimeError as e:
        raise ValueError("stresses not computable") from e


def main():

    lattice_parameter = 3.56
    lattice_structure = LatticeStructure.BCC
    species = np.array(["Cu", "Ni"])
    generator = np.random.default_rng(seed=0)

    atoms = build.bulk(
        name=species[0],
        crystalstructure=lattice_structure.name.lower(),
        a=lattice_parameter,
        cubic=True
    ).repeat((3, 3, 3))

    num_configurations = 50
    configurations = []
    for _ in range(num_configurations):
        configuration = atoms.copy()
        x_cu = generator.random()
        configuration.symbols = generator.choice(a=species, p=[x_cu, 1.0 - x_cu], size=len(configuration))

        configuration.calc = EAM(potential="Cu_Ni_Fischer_2018.eam.alloy")
        configurations.append(configuration)

    model = LimitingRidge().fit(
        configurations,
        basis=ClusterBasis(
            lattice_structure=lattice_structure,
            lattice_parameter=lattice_parameter,
            max_adjacency_order=3,
            max_triplet_order=2
        ),
        property_computer=compute_stresses,
    )

    # predict a larger stress

    supercell = Supercell(
        lattice_structure=lattice_structure,
        lattice_parameter=lattice_parameter,
        size=(10, 10, 10)
    )
    state_matrix = np.zeros((supercell.num_sites, len(species)))
    types = generator.choice(a=np.arange(len(species)), p=[0.7, 0.3], size=supercell.num_sites)
    state_matrix[np.arange(supercell.num_sites), types] = 1.0
    feature_vector = supercell.feature_vector(
        state_matrix=state_matrix,
        max_adjacency_order=3,
        max_triplet_order=2
    )
    print(feature_vector @ model.interaction_vector / supercell.num_sites)


if __name__ == "__main__":

    main()
