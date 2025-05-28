from pathlib import Path

import numpy as np

from tce.constants import LatticeStructure
from tce import topology


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

        adjacency_tensors = topology.get_adjacency_tensors_shelling(
            positions=positions,
            boxsize=boxsize,
            max_distance=4.0,
            lattice_parameter=3.59,
            lattice_structure=LatticeStructure.FCC,
            max_adjacency_order=2
        )
        three_body_tensors = topology.get_three_body_tensors(
            lattice_structure=LatticeStructure.FCC,
            adjacency_tensors=adjacency_tensors,
            max_three_body_order=1
        )

        state_matrix = np.zeros((300, 5), dtype=int)
        state_matrix[np.arange(300), types - 1] = 1
        feature_vector = topology.get_feature_vector(adjacency_tensors, three_body_tensors, state_matrix)

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

