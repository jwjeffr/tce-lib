from pathlib import Path
import re

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from tce.constants import LatticeStructure
from tce import topology


OSZICAR_PATTERN = re.compile(
    r"  ([0-9]+) F= ([+-]?(?=\.\d|\d)(?:\d+)?(?:\.?\d*))(?:[Ee]([+-]?\d+))? E0= ([+-]?(?=\.\d|\d)(?:\d+)?(?:\.?\d*))(?:[Ee]([+-]?\d+))?  d E =([+-]?(?=\.\d|\d)(?:\d+)?(?:\.?\d*))(?:[Ee]([+-]?\d+))?"
)


def main():
    root = Path("TaW_vasp_structs/MC_structs")

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

    # 16-fold cross validation
    k = 16
    sqrt_k = int(np.sqrt(k))
    assert sqrt_k == np.sqrt(k)
    scatter_fig, scatter_axs = plt.subplots(
        nrows=sqrt_k,
        ncols=sqrt_k,
        sharex=True,
        sharey=True
    )
    residuals_fig, residuals_axs = plt.subplots(
        nrows=sqrt_k,
        ncols=sqrt_k,
        sharex=True,
        sharey=True
    )
    for i in range(k):

        testing_slice = slice(len(energies) * i // k, len(energies) * (i + 1) // k)
        
        feature_vectors_training = np.delete(feature_vectors, testing_slice, axis=0)
        energies_training = np.delete(energies, testing_slice, axis=0)
        cluster_interaction_vector = np.linalg.pinv(feature_vectors_training) @ energies_training

        feature_vectors_testing = feature_vectors[testing_slice, :]
        energies_testing = energies[testing_slice]

        predicted_energies = feature_vectors_testing @ cluster_interaction_vector

        predicted_energies_training = feature_vectors_training @ cluster_interaction_vector
        ax_idx = (i % sqrt_k, i // sqrt_k)
        training_scatter = scatter_axs[ax_idx].scatter(
            energies_training, predicted_energies_training, edgecolors="black", zorder=6, alpha=0.1,
            marker="o"
        )
        testing_scatter = scatter_axs[ax_idx].scatter(
            energies_testing, predicted_energies, edgecolors="black", zorder=7, alpha=0.6,
            marker="s"
        )
        x = np.linspace(energies.min(), energies.max(), 10_000)
        scatter_axs[ax_idx].plot(x, x, linestyle="--", color="black", zorder=6)
        scatter_axs[ax_idx].tick_params(axis="x", labelrotation=45)
        scatter_axs[ax_idx].grid()

        testing_residuals = predicted_energies - energies_testing
        training_residuals = predicted_energies_training - energies_training

        training_residual = residuals_axs[ax_idx].scatter(
            energies_training, training_residuals, edgecolors="black", zorder=6, alpha=0.1,
            marker="o"
        )
        testing_residual = residuals_axs[ax_idx].scatter(
            energies_testing, testing_residuals, edgecolors="black", zorder=7, alpha=0.6,
            marker="s"
        )
        residuals_axs[ax_idx].grid()
        residuals_axs[ax_idx].tick_params(axis="x", labelrotation=45)
    
    scatter_fig.legend(
        handles=[training_scatter, testing_scatter],
        labels=["training", "testing"],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        ncols=2
    )

    scatter_fig.supxlabel("TaW DFT energy (eV)")
    scatter_fig.supylabel("TaW cluster expansion energy (eV)")
    scatter_fig.tight_layout()
    scatter_fig.savefig("figures/cross-validation.pdf", bbox_inches="tight")

    residuals_fig.legend(
        handles=[training_residual, testing_residual],
        labels=["training", "testing"],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        ncols=2
    )

    residuals_fig.supxlabel("TaW cluster expansion energy (eV)")
    residuals_fig.supylabel("Residuals (eV)")
    residuals_fig.tight_layout()
    residuals_fig.savefig("figures/residual-plot.pdf", bbox_inches="tight")


if __name__ == "__main__":

    mpl.use("TkAgg")
    main()
