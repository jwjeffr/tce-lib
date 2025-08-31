from typing import Optional, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging

import numpy as np
from ase import Atoms, build
from ase.calculators.singlepoint import SinglePointCalculator

from tce.structures import Supercell
from tce.training import CEModel


LOGGER = logging.getLogger(__name__)
rf"""logger for submodule {__name__}"""


@dataclass
class MCStep(ABC):

    r"""
    abstract base class defining a monte carlo step

    Args:
        generator (np.random.Generator): Generator instance drawing random numbers
    """

    generator: np.random.Generator

    @abstractmethod
    def step(self, state_matrix: np.typing.NDArray) -> np.typing.NDArray:

        r"""
        Method defining a Monte Carlo step. Should take in a state matrix $\mathbf{X}$, and return a new state matrix
        $\mathbf{X}'$.

        Args:
            state_matrix (np.typing.NDArray): state matrix $\mathbf{X}$
        """

        pass


class TwoParticleSwap(MCStep):

    r"""
    MC move swapping two particles
    """

    def step(self, state_matrix: np.typing.NDArray) -> np.typing.NDArray:

        r"""
        Method defining a Monte Carlo two-particle swap. Choose two sites, and swap the atoms at those sites

        Args:
            state_matrix (np.typing.NDArray): state matrix $\mathbf{X}$
        """

        new_state_matrix = state_matrix.copy()
        i, j = self.generator.integers(len(state_matrix), size=2)
        new_state_matrix[i], new_state_matrix[j] = state_matrix[j], state_matrix[i]
        return new_state_matrix


def monte_carlo(
    supercell: Supercell,
    model: CEModel,
    initial_types: np.typing.NDArray[np.integer],
    num_steps: int,
    beta: float,
    save_every: int = 1,
    generator: Optional[np.random.Generator] = None,
    mc_step: Optional[MCStep] = None,
    callback: Optional[Callable[[int, int], None]] = None
) -> list[Atoms]:

    r"""
    Monte Carlo simulation from on a lattice defined by a Supercell

    Args:
        supercell (Supercell):
            Supercell instance defining lattice
        model (CEModel):
            Container defining training data. See `tce.training.CEModel` for more info. This will usually
            be created by `tce.training.TrainingMethod.fit`.
        initial_types (np.typing.NDArray[int]):
            Initial types occupying lattice sites. This should be a 1D array of integers. For example, for a 4-site
            solid with type map defined by:
            ```py
            type_map: dict[int, str] = {0: "Fe", 1: "Cr"}
            ```
            initial types can be defined as:
            ```py
            import numpy as np

            initial_types: np.typing.NDArray[np.integer] = np.array([1, 0, 0, 1])
            ```
            which defines the first site having a Cr atom, the second site having an Fe atom, etc
        num_steps (int):
            Number of Monte Carlo steps to perform
        beta (float):
            Thermodynamic $\beta$, defined by $\beta = 1/(k_BT)$, where $k_B$ is the Boltzmann constant and $T$ is
            absolute temperature. Ensure that $k_B$ is in proper units such that $\beta$ is in appropriate units. For
            example, if the training data had energy units of eV, then $k_B$ should be defined in units of eV/K.
        save_every (int):
            How many steps to perform before saving the MC frame. This is similar to LAMMPS's `dump_every` argument
            in the `dump` command
        generator (Optional[np.random.Generator]):
            Generator instance drawing random numbers. If not specified, set to `np.random.default_rng(seed=0)`
        mc_step (Optional[MCStep]):
            Monte Carlo simulation step. If not specified, set to an instance of `TwoParticleSwap`
        callback (Optional[Callable[[int, int], None]]):
            Optional callback function that will be called after each step. Will take in the current step and the
            number of overall steps. If not specified, defaults to a call to LOGGER.info

    """

    if supercell.lattice_parameter != model.cluster_basis.lattice_parameter:
        raise ValueError(f"{supercell.lattice_parameter=} and {model.cluster_basis.lattice_parameter=} need to match!")
    if supercell.lattice_structure != model.cluster_basis.lattice_structure:
        raise ValueError(f"{supercell.lattice_structure=} and {model.cluster_basis.lattice_structure=} need to match!")

    if not generator:
        generator = np.random.default_rng(seed=0)
    if not mc_step:
        mc_step = TwoParticleSwap(generator=generator)
    if not callback:
        def callback(step_: int, num_steps_: int):
            LOGGER.info(f"MC step {step_:.0f}/{num_steps_:.0f}")

    num_types = len(model.type_map)

    ase_supercell = build.bulk(
        model.type_map[0],
        crystalstructure=supercell.lattice_structure.name.lower(),
        a=supercell.lattice_parameter,
        cubic=True,
    ).repeat(supercell.size)
    ase_supercell.symbols = model.type_map[initial_types]

    state_matrix = np.zeros((supercell.num_sites, num_types), dtype=int)
    state_matrix[np.arange(supercell.num_sites), initial_types] = 1

    trajectory = []
    energy = model.interaction_vector @ supercell.feature_vector(
        state_matrix=state_matrix,
        max_adjacency_order=model.cluster_basis.max_adjacency_order,
        max_triplet_order=model.cluster_basis.max_triplet_order
    )
    for step in range(num_steps):

        # LOGGER.info(f"MC step {step:.0f}/{num_steps:.0f}")
        callback(step, num_steps)

        if not step % save_every:
            _, types = np.where(state_matrix)
            atoms = ase_supercell.copy()
            atoms.set_chemical_symbols(symbols=model.type_map[types])
            atoms.calc = SinglePointCalculator(atoms=atoms, energy=energy)
            trajectory.append(atoms)
            LOGGER.info(f"saved configuration at step {step:.0f}/{num_steps:.0f}")

        new_state_matrix = mc_step.step(state_matrix)
        feature_diff = supercell.clever_feature_diff(
            state_matrix, new_state_matrix,
            max_adjacency_order=model.cluster_basis.max_adjacency_order,
            max_triplet_order=model.cluster_basis.max_triplet_order
        )
        energy_diff = model.interaction_vector @ feature_diff
        if np.exp(-beta * energy_diff) > 1.0 - generator.random():
            LOGGER.info(f"move accepted with energy difference {energy_diff:.3f}")
            state_matrix = new_state_matrix
            energy += energy_diff

    return trajectory