from itertools import product
from string import Template
from pathlib import Path
from io import StringIO

import numpy as np

from tce.structures import Supercell
from tce.constants import LatticeStructure


def main():

    supercell = Supercell(
        lattice_structure=LatticeStructure.BCC,
        lattice_parameter=2.87,
        size=(5, 5, 5)
    )

    template_path = Path("data-file-template.txt")
    with template_path.open("r") as file:
        template = Template(file.read())
    samples_dir = Path("lammps-data/samples")

    low, high, grid_spacing = 0.1, 0.9, 15
    grid = np.linspace(low, high, grid_spacing)

    generator = np.random.default_rng(seed=0)
    for composition in product(grid, repeat=4):
        composition = (1.0 - sum(composition), *composition)
        if not all(x >= low for x in composition):
            continue
        types = generator.choice(np.arange(5), p=composition, size=supercell.num_sites)

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
        masses = [55.845, 51.9961]
        data_str = template.substitute(
            {
                "header": f"generated from {template_path}",
                "num_atoms": supercell.num_sites,
                "num_types": len(masses),
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
        rand = generator.integers(9999)
        data_dir = samples_dir / Path(f"{rand:.0f}")
        data_dir.mkdir(exist_ok=True)
        with (data_dir / "unrelaxed.dat").open("w") as file:
            file.write(data_str)


if __name__ == "__main__":

    main()
