from pathlib import Path
import logging
import sys

import numpy as np
from ase import io

from tce.structures import Supercell
from tce.training import CEModel
from tce.monte_carlo import monte_carlo


def main():

    rng = np.random.default_rng(seed=0)

    model = CEModel.load(Path("CuNi.npz"))

    supercell = Supercell(
        lattice_structure=model.cluster_basis.lattice_structure,
        lattice_parameter=model.cluster_basis.lattice_parameter,
        size=(10, 10, 10)
    )

    trajectory = monte_carlo(
        supercell=supercell,
        model=model,
        initial_types=rng.integers(len(model.type_map), size=supercell.num_sites),
        num_steps=10_000,
        beta=19.341,
        save_every=100
    )

    for i, frame in enumerate(trajectory):
        path = Path(f"copper-nickel/frame_{i:.0f}.xyz")
        path.parent.mkdir(parents=True, exist_ok=True)
        io.write(path, frame)


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    main()
