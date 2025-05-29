from pathlib import Path

import numpy as np
from ase.io import lammpsdata
import sparse
import matplotlib.pyplot as plt
import matplotlib as mpl

from tce.constants import LatticeStructure, STRUCTURE_TO_CUTOFF_LISTS
from tce import topology


def main():
    root = Path("lammps-data")
    lattice_parameter = 2.5
    cutoffs = [lattice_parameter * x for x in STRUCTURE_TO_CUTOFF_LISTS[LatticeStructure.IDEAL_HCP][1:4]]
    tolerance = 0.01

    feature_vectors = []
    energies = []
    mg_concentrations = []
    for directory in root.glob("samples/*"):

        energy = np.loadtxt(directory / "energy.txt")
        energies.append(energy)

        data = lammpsdata.read_lammps_data(directory / "unrelaxed.dat")
        state_matrix = np.zeros((len(data), 2))
        for i, symbol in enumerate(data.get_chemical_symbols()):
            if symbol == "Fe":
                state_matrix[i, 0] = 1
            elif symbol == "Mg":
                state_matrix[i, 1] = 1
            else:
                raise ValueError
        mg_concentrations.append(state_matrix[:, 1].mean())
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

    mg_concentrations = np.array(mg_concentrations)
    feature_vectors = np.array(feature_vectors)
    energies = np.array(energies)
    
    cluster_interaction_vector = np.linalg.pinv(feature_vectors) @ energies
    predicted_energies = feature_vectors @ cluster_interaction_vector
    inaccuracy = np.linalg.norm(energies - predicted_energies)
    inaccuracy /= np.linalg.norm(energies - energies.mean())

    np.savetxt("MgZn.txt", cluster_interaction_vector, fmt="%.18f")

    # calculate energy for pure iron and pure mg of the same size
    pure_iron_state_matrix = np.zeros((len(data), 2))
    pure_iron_state_matrix[:, 0] = 1.0
    pure_iron_energy = cluster_interaction_vector @ topology.get_feature_vector(
        adjacency_tensors,
        three_body_tensors,
        pure_iron_state_matrix
    )
    pure_mg_state_matrix = np.zeros((len(data), 2))
    pure_mg_state_matrix[:, 1] = 1.0
    pure_mg_energy = cluster_interaction_vector @ topology.get_feature_vector(
        adjacency_tensors,
        three_body_tensors,
        pure_mg_state_matrix
    )

    energy_of_mixing = energies - mg_concentrations * pure_mg_energy - (1.0 - mg_concentrations) * pure_iron_energy
    energy_of_mixing /= len(data)

    plt.scatter(100.0 * mg_concentrations, energy_of_mixing, alpha=0.6, edgecolors="black", zorder=7, color="magenta")
    plt.grid()
    plt.xlabel("Mg concentration (at. %)")
    plt.ylabel("Energy of mixing (eV/atom)")
    ax = plt.gca()
    secax = ax.secondary_xaxis("top", functions=(lambda x: 100 - x, lambda x: 100 - x))
    secax.set_xlabel("Fe concentration (at. %)")
    plt.tight_layout()
    plt.savefig("mixing.pdf", bbox_inches="tight")


if __name__ == "__main__":

    mpl.use("TkAgg")
    main()

