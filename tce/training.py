from dataclasses import dataclass
from typing import Optional, Sequence
from datetime import datetime
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from scipy.spatial import KDTree
from ase import Atoms

from tce.constants import LatticeStructure, STRUCTURE_TO_CUTOFF_LISTS
from tce.topology import get_adjacency_tensors, get_three_body_tensors, get_feature_vector


@dataclass
class TrainingMethod(ABC):

    r"""
    Abstract base class for defining how to train a model $y = \beta^\intercal X$. $X$ here is **not** a state matrix,
    but rather a data matrix.
    """

    @abstractmethod
    def train(
        self,
        X: np.typing.NDArray[np.floating],
        y: np.typing.NDArray[np.floating]
    ) -> np.typing.NDArray[np.floating]:

        r"""
        Train method for model training.

        Args:
            X (np.typing.NDArray[float]):
                The data matrix $X$.
            y (np.typing.typing.NDArray[float]):
                The target vector $y$.
        """

        pass


class LimitingRidge(TrainingMethod):

    r"""
    Train by minimizing the limiting ridge problem:

    $$ L(\beta\;|\;\lambda) = \|X\beta - y \|_2^2 + \lambda \|\beta\|_2^2 $$

    $$ \hat{\beta} = \lim_{\lambda\to 0^+}\arg\min_{\beta} L(\beta\;|\;\lambda) = X^+ y $$

    where $X^+$ denotes the [Moore-Penrose inverse](https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse).
    """

    def train(
        self,
        X: np.typing.NDArray[np.floating],
        y: np.typing.NDArray[np.floating]
    ) -> np.typing.NDArray[np.floating]:

        r"""
        Train via the limiting ridge problem. Largely an alias for $\hat{\beta} = X^+ y$

        Args:
            X (np.typing.NDArray[float]):
                The data matrix $X$.
            y (np.typing.typing.NDArray[float]):
                The target vector $y$.
        """

        return np.linalg.pinv(X) @ y


@dataclass
class TrainingContainer:

    r"""
    Object containing useful information from training. Largely a convenience class
    """

    lattice_structure: LatticeStructure
    r"""lattice structure that the trained model corresponds to"""

    lattice_parameter: float
    r"""lattice parameter that the trained model corresponds to"""

    max_adjacency_order: int
    r"""maximum adjacency order (number of nearest neighbors) that the trained model accounts for"""

    max_triplet_order: int
    r"""maximum triplet order (number of three-body clusters) that the trained model accounts for"""

    interaction_vector: np.typing.NDArray[np.floating]
    r"""trained interaction vector"""

    type_map: np.typing.NDArray[np.str_]
    r"""array of chemical species, e.g. `np.array(["Fe", "Cr"])`"""

    datetime: datetime
    r"""datetime that the model was trained"""

    description: Optional[str] = None
    r"""optional container description"""

    author: Optional[str] = None
    r"""optional author name"""

    def to_npz(self, path: Path) -> None:

        r"""
        save container as an npz object

        Args:
            path (Path):
                Path to save the resulting npz file.
        """

        metadata = {
            "datetime": self.datetime.isoformat(),
            "lattice_structure": self.lattice_structure.name.lower(),
            "lattice_parameter": self.lattice_parameter,
            "max_adjacency_order": self.max_adjacency_order,
            "max_triplet_order": self.max_triplet_order,
            "type_map": self.type_map
        }

        if self.description:
            metadata["description"] = self.description
        if self.author:
            metadata["author"] = self.author

        with path.open("wb") as file:
            np.savez(file, interaction_vector=self.interaction_vector, **metadata) # type: ignore [arg-type]

    @classmethod
    def from_npz(cls, path: Path) -> "TrainingContainer":

        r"""
        Load container from npz file.

        Args:
            path (Path):
                Where to load the container from.
        """

        training_data = np.load(path)

        metadata = {
            "lattice_structure": getattr(LatticeStructure, training_data["lattice_structure"].item().upper()),
            "lattice_parameter": training_data["lattice_parameter"].item(),
            "max_adjacency_order": training_data["max_adjacency_order"].item(),
            "max_triplet_order": training_data["max_triplet_order"].item(),
            "type_map": training_data["type_map"],
            "datetime": datetime.fromisoformat(training_data["datetime"].item())
        }

        if "description" in training_data:
            metadata["description"] = training_data["description"].item()
        if "author" in training_data:
            metadata["author"] = training_data["author"].item()

        return cls(interaction_vector=training_data["interaction_vector"], **metadata)

    @classmethod
    def from_ase_atoms(
        cls,
        configurations: Sequence[Atoms],
        lattice_parameter: float,
        lattice_structure: LatticeStructure,
        max_adjacency_order: int,
        max_triplet_order: int,
        type_map: np.typing.NDArray[np.str_],
        training_method: Optional[TrainingMethod] = None,
        description: Optional[str] = None,
        author: Optional[str] = None
    ):

        """
        Load a training container from a sequence of `ase.Atoms` objects, equivalently train on configurations defined
        by `ase`. Each `ase.Atoms` object needs a calculator to compute the total energy - see the documentation for
        this [here](https://ase-lib.org/ase/calculators/calculators.html).

        Args:
            configurations (Sequence[ase.Atoms]):
                Sequence of `ase.Atoms` objects defining atomic configurations.
            lattice_parameter (float):
                The lattice parameter of the configurations. It is expected that each configuration has the same
                lattice parameter. This is not checked within the call, so be very careful! (**TODO**)
            lattice_structure (LatticeStructure):
                The lattice structure of the configurations. It is expected that each configuration has the same
                lattice structure. This is not checked within the call, so be very careful! (**TODO**)
            max_adjacency_order (int):
                maximum adjacency order (number of nearest neighbors) that the trained model accounts for
            max_triplet_order (int):
                maximum triplet order (number of three-body clusters) that the trained model accounts for
            type_map (np.typing.NDArray[str]):
                array of chemical species, e.g. `np.array(["Fe", "Cr"])`
            training_method (TrainingMethod):
                training method to train the model. If not specified, set to an instance of `LimitingRidge`
            description (str):
                optional container description
            author (str):
                optional author name
        """

        if not training_method:
            training_method = LimitingRidge()

        num_types = len(type_map)
        inverse_type_map = {symbol: i for i, symbol in enumerate(type_map)}

        feature_size = max_adjacency_order * num_types ** 2 + max_triplet_order * num_types ** 3
        X = np.zeros((len(configurations), feature_size))
        y = np.zeros(len(configurations))

        for index, atoms in enumerate(configurations):
            tree = KDTree(atoms.positions, boxsize=np.diag(atoms.cell))
            adjacency_tensors = get_adjacency_tensors(
                tree=tree,
                cutoffs=lattice_parameter * STRUCTURE_TO_CUTOFF_LISTS[lattice_structure][:max_adjacency_order],
            )
            three_body_tensors = get_three_body_tensors(
                lattice_structure=lattice_structure,
                adjacency_tensors=adjacency_tensors,
                max_three_body_order=max_triplet_order,
            )

            state_matrix = np.zeros((len(atoms), num_types))
            for site, symbol in enumerate(atoms.symbols):
                state_matrix[site, inverse_type_map[symbol]] = 1.0

            # compute the feature vector and store it
            X[index, :] = get_feature_vector(
                adjacency_tensors=adjacency_tensors,
                three_body_tensors=three_body_tensors,
                state_matrix=state_matrix
            )

            y[index] = atoms.get_potential_energy()

        return cls(
            lattice_structure=lattice_structure,
            lattice_parameter=lattice_parameter,
            max_adjacency_order=max_adjacency_order,
            max_triplet_order=max_triplet_order,
            interaction_vector=training_method.train(X=X, y=y),
            type_map=type_map,
            datetime=datetime.now(),
            description=description,
            author=author
        )
