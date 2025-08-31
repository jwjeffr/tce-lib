from pathlib import Path
import os

import numpy as np
import requests

from tce.structures import Supercell
from tce.training import CEModel
from tce.monte_carlo import monte_carlo


def discord_webhook_callback(
        step: int,
        num_steps: int,
        env_var: str = "DISCORD_WEBHOOK_URL",
        message: str = "CuNi mc run finished"
):

    if not os.getenv(env_var):
        raise ValueError(f"please set {env_var} environment variable")

    if step + 1 < num_steps:
        return

    response = requests.post(url=os.getenv(env_var), json={"content": message})
    response.raise_for_status()


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
        save_every=100,
        callback=discord_webhook_callback,
    )

if __name__ == "__main__":

    main()
