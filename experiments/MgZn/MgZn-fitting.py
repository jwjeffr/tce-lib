from pathlib import Path

import numpy as np
from ase.io import lammpsdata
import sparse

from tce.constants import LatticeStructure, STRUCTURE_TO_CUTOFF_LISTS
from tce import topology


def main():
    root = Path("lammps-data")
    lattice_parameter = 3.15
    cutoffs = [lattice_parameter * x for x in STRUCTURE_TO_CUTOFF_LISTS[LatticeStructure.IDEAL_HCP][1:4]]
    tolerance = 0.01

    feature_vectors = []
    energies = []
    for directory in root.glob("samples/*"):

        energy = np.loadtxt(directory / "energy.txt")
        energies.append(energy)

        data = lammpsdata.read_lammps_data(directory / "unrelaxed.dat")
        state_matrix = np.zeros((len(data), 2))
        for i, symbol in enumerate(data.get_chemical_symbols()):
            if symbol == "Mg":
                state_matrix[i, 0] = 1
            elif symbol == "Zn":
                state_matrix[i, 1] = 1
            else:
                raise ValueError
        distances = data.get_all_distances(mic=True)
        adjacency_tensors = sparse.stack([
            sparse.COO.from_numpy(
                np.logical_and((1.0 - tolerance) * cutoff < distances, distances < (1.0 + tolerance) * cutoff)
            ) for cutoff in cutoffs
        ])
        three_body_tensors = topology.get_three_body_tensors(
            lattice_structure=LatticeStructure.IDEAL_HCP,
            adjacency_tensors=adjacency_tensors,
            max_three_body_order=2
        )

        feature_vectors.append(topology.get_feature_vector(
            adjacency_tensors=adjacency_tensors,
            three_body_tensors=three_body_tensors,
            state_matrix=state_matrix
        ))

    feature_vectors = np.array(feature_vectors)
    energies = np.array(energies)
    
    cluster_interaction_vector = np.linalg.pinv(feature_vectors) @ energies
    predicted_energies = feature_vectors @ cluster_interaction_vector
    inaccuracy = np.linalg.norm(energies - predicted_energies)
    inaccuracy /= np.linalg.norm(energies - energies.mean())

    np.savetxt("MgZn.txt", cluster_interaction_vector, fmt="%.18f")


if __name__ == "__main__":

    main()

