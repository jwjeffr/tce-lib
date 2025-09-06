from pathlib import Path
import logging
import sys
from typing import Callable
from functools import wraps

import numpy as np
import matplotlib.pyplot as plt
from ase import build

from tce.training import ClusterExpansion
from tce.monte_carlo import monte_carlo, MCStep


def one_particle_swap_factory(generator: np.random.Generator) -> MCStep:

    @wraps(one_particle_swap_factory)
    def wrapper(state_matrix: np.typing.NDArray[np.floating]) -> np.typing.NDArray[np.floating]:
        num_sites, num_types = state_matrix.shape
        i, = generator.integers(num_sites, size=1)
        current_type = np.where(state_matrix[i, :] == 1)[0]

        new_type = generator.integers(num_types, size=1)
        while new_type == current_type:
            new_type = generator.integers(num_types, size=1)

        new_state_matrix = state_matrix.copy()
        new_state_matrix[i, :] = np.zeros(num_types)
        new_state_matrix[i, new_type] = 1.0
        return new_state_matrix

    return wrapper


def energy_modifier_factory(
    chemical_potentials: np.typing.NDArray[np.floating]
) -> Callable[[np.typing.NDArray[np.floating], np.typing.NDArray[np.floating]], float]:

    @wraps(energy_modifier_factory)
    def wrapper(
        state_matrix: np.typing.NDArray[np.floating],
        new_state_matrix: np.typing.NDArray[np.floating]
    ) -> float:
        change_in_num_types = new_state_matrix.sum(axis=0) - state_matrix.sum(axis=0)
        return -chemical_potentials @ change_in_num_types

    return wrapper


def main():

    rng = np.random.default_rng(seed=0)

    cluster_expansion = ClusterExpansion.load(Path("CuNi.pkl"))

    chemical_potentials_cu = np.linspace(-0.5, 1.5, 25)
    atomic_fractions_cu = np.zeros_like(chemical_potentials_cu)

    pure_ni = build.bulk(
        cluster_expansion.type_map[1],
        a=cluster_expansion.cluster_basis.lattice_parameter,
        crystalstructure=cluster_expansion.cluster_basis.lattice_structure.name.lower(),
        cubic=True
    ).repeat((10, 10, 10))

    for i, chemical_potential_cu in enumerate(chemical_potentials_cu):

        trajectory = monte_carlo(
            initial_configuration=pure_ni,
            cluster_expansion=cluster_expansion,
            num_steps=10_000,
            beta=19.341,
            save_every=1_000,
            energy_modifier=energy_modifier_factory(
                chemical_potentials=np.array([chemical_potential_cu, 0.0])
            ),
            mc_step=one_particle_swap_factory(generator=rng),
            callback=lambda x, y: None
        )
        final_types = np.array(trajectory[-1].get_chemical_symbols())
        atomic_fractions_cu[i] = (final_types == "Cu").mean()

    plt.plot(chemical_potentials_cu, 100 * atomic_fractions_cu, color="orangered")
    plt.xlabel(r"$\mu_\text{Cu} - \mu_\text{Ni}$ (eV)")
    plt.ylabel(r"Cu concentration (at. %)")
    plt.grid()
    plt.tight_layout()
    plt.savefig("cu-ni-sgcmc.png", dpi=800, bbox_inches="tight")


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    main()
