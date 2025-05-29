import numpy as np
from io import StringIO
from pathlib import Path
from string import Template

from tce.structures import Supercell
from tce.constants import LatticeStructure


def main():

    num_types = 5
    supercell = Supercell(
        lattice_parameter=3.59,
        lattice_structure=LatticeStructure.FCC,
        size=(10, 10, 10)
    )
    interaction_vector = np.loadtxt("CoNiCrFeMn-interaction-vector.txt")
    generator = np.random.default_rng(seed=0)
    types = generator.integers(num_types, size=supercell.num_sites)
    state_matrix = np.zeros((supercell.num_sites, num_types), dtype=int)
    state_matrix[np.arange(supercell.num_sites), types] = 1

    beta = 38.68

    num_steps = 1_000_000
    dump_every = 10_000

    template_path = Path("data-file-template.txt")
    with template_path.open("r") as file:
        template = Template(file.read())

    for step in range(num_steps):

        if not (step + 1) % dump_every:

            print(step + 1)

            _, types = np.where(state_matrix == 1)

            with StringIO() as s:
                position_data = np.vstack(
                    (np.arange(1, supercell.num_sites + 1), types + 1, *supercell.positions.T,
                     *np.zeros_like(supercell.positions.T))
                ).T
                np.savetxt(s, position_data, fmt=("%.0f", "%.0f", "%.3f", "%.3f", "%.3f", "%.0f", "%.0f", "%.0f"))
                s.seek(0)
                position_data = s.read()
            with StringIO() as s:
                velocity_data = np.vstack(
                    (np.arange(1, supercell.num_sites + 1), *np.zeros_like(supercell.positions.T))
                ).T
                np.savetxt(s, velocity_data, fmt="%.0f")
                s.seek(0)
                velocity_data = s.read()

            boxsize = supercell.lattice_parameter * np.array(supercell.size)
            masses = [58.933, 58.69, 51.996, 55.847, 54.94]
            data_str = template.substitute(
                {
                    "header": f"generated from {template_path}",
                    "num_atoms": supercell.num_sites,
                    "num_types": 5,
                    "xlo": 0.0,
                    "xhi": boxsize[0],
                    "ylo": 0.0,
                    "yhi": boxsize[1],
                    "zlo": 0.0,
                    "zhi": boxsize[2],
                    "masses": "\n".join(f"{i + 1:.0f} {mass:.3f}" for i, mass in enumerate(masses)),
                    "position_data": position_data,
                    "velocity_data": velocity_data
                }
            )

            with (Path("sim") / f"cluster_expansion-{step + 1:.0f}.dat").open("w") as file:
                file.write(data_str)

        new_state_matrix = state_matrix.copy()
        i, j = generator.integers(supercell.num_sites, size=2)
        new_state_matrix[i], new_state_matrix[j] = state_matrix[j], state_matrix[i]
        feature_diff = supercell.clever_feature_diff(
            state_matrix, new_state_matrix,
            max_adjacency_order=2, max_triplet_order=1
        )
        energy_diff = np.dot(interaction_vector, feature_diff)
        if np.exp(-beta * energy_diff) > 1.0 - generator.random():
            state_matrix = new_state_matrix


if __name__ == "__main__":

    main()
