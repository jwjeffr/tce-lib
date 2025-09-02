from pathlib import Path

from ase.io.lammpsdata import read_lammps_data
from ase.io import write
from ase.calculators.singlepoint import SinglePointCalculator
import numpy as np


def main():

    paths = list(Path("/home/jwjeffr/tce-testing/experiments/surrogate-CoNiCrFeMn/structure_data").iterdir())

    for i, p in enumerate(paths):
        atoms = read_lammps_data(p / "configuration.dat")
        energy = np.loadtxt(p / "energy.txt")

        atoms.calc = SinglePointCalculator(atoms, energy=energy)
        write(f"configuration_{i:.0f}.xyz", atoms)


if __name__ == "__main__":

    main()
