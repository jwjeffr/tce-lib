from pathlib import Path
import re

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.spatial import ConvexHull
from matplotlib.collections import LineCollection

from tce.constants import LatticeStructure
from tce import topology, structures

OSZICAR_PATTERN = re.compile(
    r"  ([0-9]+) F= ([+-]?(?=\.\d|\d)(?:\d+)?(?:\.?\d*))(?:[Ee]([+-]?\d+))? E0= ([+-]?(?=\.\d|\d)(?:\d+)?(?:\.?\d*))(?:[Ee]([+-]?\d+))?  d E =([+-]?(?=\.\d|\d)(?:\d+)?(?:\.?\d*))(?:[Ee]([+-]?\d+))?"
)


def main():
    root = Path("vasp-structures/MC_structs")

    feature_vectors = []
    energies = []
    for directory in root.glob("*_Ta*W*"):

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

        adjacency_tensors = topology.get_adjacency_tensors_binning(
            positions=positions,
            boxsize=np.diag(cell_matrix),
            max_distance=4.0,
            max_adjacency_order=2
        )

        three_body_tensors = topology.get_three_body_tensors(
            lattice_structure=LatticeStructure.BCC,
            adjacency_tensors=adjacency_tensors,
            max_three_body_order=1
        )

        state_matrix = np.zeros((128, 2), dtype=int)
        state_matrix[np.arange(128), types] = 1

        feature_vector = topology.get_feature_vector(
            adjacency_tensors=adjacency_tensors,
            three_body_tensors=three_body_tensors,
            state_matrix=state_matrix
        )

        feature_vectors.append(feature_vector)

    feature_vectors = np.array(feature_vectors)
    energies = np.array(energies)

    interaction_vector = np.linalg.pinv(feature_vectors) @ energies

    generator = np.random.default_rng(seed=0)
    supercell = structures.Supercell(
        lattice_structure=LatticeStructure.BCC,
        lattice_parameter=1.0,
        size=(10, 10, 10)
    )
    tungsten_concentrations = np.linspace(0.0, 1.0, 100)
    tungsten_concentrations = np.repeat(tungsten_concentrations, 25)
    interaction_energies = np.zeros_like(tungsten_concentrations)
    for i, x_tungsten in enumerate(tungsten_concentrations):

        types = generator.choice((0, 1), p=(1.0 - x_tungsten, x_tungsten), size=supercell.num_sites)
        state_matrix = np.zeros((supercell.num_sites, 2), dtype=int)
        state_matrix[np.arange(supercell.num_sites), types] = 1
        feature_vector = supercell.feature_vector(state_matrix=state_matrix, max_adjacency_order=2, max_triplet_order=1)
        interaction_energies[i] = interaction_vector @ feature_vector

    enthalpy_of_mixing = interaction_energies - \
        tungsten_concentrations * interaction_energies[-1] - (1.0 - tungsten_concentrations) * interaction_energies[0]
    enthalpy_of_mixing /= supercell.num_sites

    tungsten_concentrations *= 100
    points = np.column_stack((tungsten_concentrations, enthalpy_of_mixing))
    hull = ConvexHull(points)

    y_values = points[hull.simplices, 1]

    mask = np.all(y_values <= 0, axis=1)
    valid_simplices = hull.simplices[mask]
    segments = points[valid_simplices]

    plt.scatter(tungsten_concentrations, enthalpy_of_mixing, alpha=0.3, edgecolors="black", zorder=7, label="cluster expansion")
    plt.gca().add_collection(
        LineCollection(segments, colors="black", linestyles="-", zorder=8, label="convex hull")
    )
    plt.xlabel("W Concentration (at. %)")
    plt.ylabel("Enthalpy of mixing (eV / atom)")
    ax = plt.gca()
    secax = ax.secondary_xaxis(location="top", functions=(lambda x: 100 - x, lambda x: 100 - x))
    secax.set_xlabel("Ta Concentration (at. %)")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig("convex-hull.pdf", bbox_inches="tight")


if __name__ == "__main__":
    mpl.use("TkAgg")
    main()
