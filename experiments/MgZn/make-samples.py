from pathlib import Path

from ase.build import bulk
from ase.io.lammpsdata import write_lammps_data
import numpy as np


def main():

    a = 3.15
    c = np.sqrt(8.0 / 3.0) * a
    rng = np.random.default_rng(seed=0)

    for x_fe in np.linspace(0.001, 0.999, 500):
        supercell = bulk('Mg', 'hcp', a=a, c=c).repeat((5, 5, 4))

        num_atoms = len(supercell)
        fe_indices = rng.choice(num_atoms, size=int(x_fe * num_atoms), replace=False)

        for i in fe_indices:
            supercell[i].symbol = 'Zn'

        sample_dir = Path(f"lammps-data/samples/{rng.integers(9999):.0f}")
        sample_dir.mkdir(exist_ok=True)
        write_lammps_data(sample_dir / "unrelaxed.dat", supercell, atom_style='atomic', masses=True)


if __name__ == "__main__":

    main()
