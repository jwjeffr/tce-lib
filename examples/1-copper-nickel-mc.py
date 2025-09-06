from pathlib import Path
import logging
import sys

import numpy as np
from ase import io, build

from tce.training import ClusterExpansion
from tce.monte_carlo import monte_carlo


def main():

    rng = np.random.default_rng(seed=0)

    cluster_expansion = ClusterExpansion.load(Path("CuNi.pkl"))

    atoms = build.bulk(
        cluster_expansion.type_map[0],
        a=cluster_expansion.cluster_basis.lattice_parameter,
        crystalstructure=cluster_expansion.cluster_basis.lattice_structure.name.lower(),
        cubic=True
    ).repeat((10, 10, 10))
    atoms.symbols = rng.choice(cluster_expansion.type_map, size=len(atoms))

    trajectory = monte_carlo(
        initial_configuration=atoms,
        cluster_expansion=cluster_expansion,
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
