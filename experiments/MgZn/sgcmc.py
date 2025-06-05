from pathlib import Path

import numpy as np
from ase.build import bulk
from ase.io import lammpsdata
import sparse
import matplotlib.pyplot as plt
import matplotlib as mpl

from tce import topology
from tce.constants import LatticeStructure, STRUCTURE_TO_CUTOFF_LISTS


def main():
    a = 3.15
    cutoffs = [a * x for x in STRUCTURE_TO_CUTOFF_LISTS[LatticeStructure.IDEAL_HCP][1:4]]
    c = np.sqrt(8.0 / 3.0) * a
    tolerance = 0.01
    temperature = 400
    beta = 1.0 / (8.617e-5 * temperature)
    interaction_vector = np.loadtxt("MgZn.txt")

    num_steps = 100_000
    rng = np.random.default_rng(seed=0)

    supercell = bulk('Mg', 'hcp', a=a, c=c).repeat((15, 10, 10))
    distances = supercell.get_all_distances(mic=True)

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

    chemical_potentials_zn = np.linspace(-0.7, -0.1, 25)
    mg_concentration_first = np.zeros_like(chemical_potentials_zn)

    dump_dir = Path("hysteresis")
    dump_dir.mkdir(exist_ok=True)
    type_map = ["Mg", "Zn"]
    for i, mu_zn in enumerate(chemical_potentials_zn):

        print(i)

        state_matrix = np.zeros((len(supercell), 2))
        state_matrix[:, 0] = 1.0
        chemical_potentials = np.array([0.0, mu_zn])

        for step in range(num_steps):

            if not step % 1_000:
                print(step)

            site = rng.integers(len(supercell))
            new_state_matrix = state_matrix.copy()
            new_state_matrix[site, :] = 1.0 - new_state_matrix[site, :]

            mu_init = chemical_potentials[np.where(state_matrix[site, :] == 1)]
            mu_final = chemical_potentials[np.where(new_state_matrix[site, :] == 1)]

            change_in_energy = mu_init - mu_final + interaction_vector @ topology.get_feature_vector_difference(
                adjacency_tensors=adjacency_tensors,
                three_body_tensors=three_body_tensors,
                initial_state_matrix=state_matrix,
                final_state_matrix=new_state_matrix
            )

            if np.exp(-beta * change_in_energy) > 1.0 - rng.random():
                state_matrix = new_state_matrix

        print()
        _, types = np.where(state_matrix)
        types = (type_map[t] for t in types)
        supercell.set_chemical_symbols(types)
        lammpsdata.write_lammps_data(dump_dir / f"first-mc-{i:.0f}.dat", supercell, masses=True, specorder=type_map)
        mg_concentration_first[i] = state_matrix[:, 0].mean()

    mg_concentration_second = np.zeros_like(chemical_potentials_zn)
    for i, mu_zn in enumerate(chemical_potentials_zn):

        print(i)

        state_matrix = np.zeros((len(supercell), 2))
        state_matrix[:, 1] = 1.0
        chemical_potentials = np.array([0.0, mu_zn])

        for step in range(num_steps):

            if not step % 1_000:

                print(step)

            site = rng.integers(len(supercell))
            new_state_matrix = state_matrix.copy()
            new_state_matrix[site, :] = 1.0 - new_state_matrix[site, :]

            mu_init = chemical_potentials[np.where(state_matrix[site, :] == 1)]
            mu_final = chemical_potentials[np.where(new_state_matrix[site, :] == 1)]

            change_in_energy = mu_init - mu_final + interaction_vector @ topology.get_feature_vector_difference(
                adjacency_tensors=adjacency_tensors,
                three_body_tensors=three_body_tensors,
                initial_state_matrix=state_matrix,
                final_state_matrix=new_state_matrix
            )

            if np.exp(-beta * change_in_energy) > 1.0 - rng.random():
                state_matrix = new_state_matrix

        print()
        _, types = np.where(state_matrix)
        types = (type_map[t] for t in types)
        supercell.set_chemical_symbols(types)
        lammpsdata.write_lammps_data(dump_dir / f"second-mc-{i:.0f}.dat", supercell, masses=True, specorder=type_map)
        mg_concentration_second[i] = state_matrix[:, 0].mean()

    plt.plot(chemical_potentials_zn, 100 * mg_concentration_first)
    plt.plot(chemical_potentials_zn, 100 * mg_concentration_second)
    plt.xlabel(r"$\mu_\text{Zn} - \mu_\text{Mg}$ (eV)")
    plt.ylabel("Mg concentration (at. %)")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"hysteresis_T{temperature:.0f}.pdf", bbox_inches="tight")


if __name__ == "__main__":

    mpl.use("TkAgg")
    main()
