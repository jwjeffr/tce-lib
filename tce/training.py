from dataclasses import dataclass
from abc import abstractmethod
from typing import Callable, TypeAlias, Union, Optional, Protocol, runtime_checkable, Self
import warnings
from pathlib import Path
import pickle

import numpy as np
from ase import Atoms

from tce.constants import ClusterBasis, STRUCTURE_TO_ATOMIC_BASIS
from tce.topology import FeatureComputer, topological_feature_vector_factory


NON_CUBIC_CELL_MESSAGE = "At least one of your configurations has a non-cubic cell. For now, tce-lib does not support non-cubic lattices."

INCOMPATIBLE_GEOMETRY_MESSAGE = "Geometry in all configurations must match geometry in cluster basis."

NO_POTENTIAL_ENERGY_MESSAGE = "At least one of your configurations does not have a computable potential energy."

LARGE_SYSTEM_THRESHOLD = 1_000
LARGE_SYSTEM_MESSAGE = f"You have passed a relatively large system (larger than {LARGE_SYSTEM_THRESHOLD:.0f}) as a training point. This will be very slow."


def get_type_map(configurations: list[Atoms]) -> np.typing.NDArray[np.str_]:

    # not all configurations need to have the same number of types, calculate the union of types
    all_types = set.union(*(set(x.get_chemical_symbols()) for x in configurations))
    return np.array(sorted(list(all_types)))


PropertyComputer: TypeAlias = Callable[[Atoms], Union[float, np.typing.NDArray[np.floating]]]

def total_energy(atoms: Atoms) -> float:

    try:
        return atoms.get_potential_energy()
    except RuntimeError as e:
        raise ValueError(NO_POTENTIAL_ENERGY_MESSAGE) from e


def get_data_pairs(
    configurations: list[Atoms],
    basis: ClusterBasis,
    target_property_computer: PropertyComputer,
    feature_computer: FeatureComputer,
) -> tuple[np.typing.NDArray[np.floating], np.typing.NDArray[np.floating]]:

    basis_atomic_volume = basis.lattice_parameter ** 3 / len(STRUCTURE_TO_ATOMIC_BASIS[basis.lattice_structure])
    for configuration in configurations:

        if np.any(configuration.get_cell().angles() != 90.0 * np.ones(3)):
            raise ValueError(NON_CUBIC_CELL_MESSAGE)

        configuration_atomic_volume = configuration.get_volume() / len(configuration)
        if not np.isclose(configuration_atomic_volume, basis_atomic_volume):
            raise ValueError(INCOMPATIBLE_GEOMETRY_MESSAGE)

        if len(configuration) > LARGE_SYSTEM_THRESHOLD:
            warnings.warn(LARGE_SYSTEM_MESSAGE, UserWarning)

    type_map = get_type_map(configurations)
    num_types = len(type_map)

    feature_size = basis.max_adjacency_order * num_types ** 2 + basis.max_triplet_order * num_types ** 3
    X = np.zeros((len(configurations), feature_size))
    y: list[Union[float, np.typing.NDArray[np.floating]]] = [np.nan] * len(configurations)

    for index, atoms in enumerate(configurations):

        y[index] = target_property_computer(atoms)
        X[index, :] = feature_computer(atoms)

    return X, np.array(y)


@runtime_checkable
class Model(Protocol):

    @abstractmethod
    def fit(self, X: np.typing.NDArray[np.floating], y: np.typing.NDArray[np.floating]) -> Self:

        pass

    @abstractmethod
    def predict(self, x: np.typing.NDArray[np.floating]) -> Union[np.typing.NDArray[np.floating], float]:

        pass


class LimitingRidge:

    def fit(self, X: np.typing.NDArray[np.floating], y: np.typing.NDArray[np.floating]) -> Self:

        self.coef_ = np.linalg.pinv(X) @ y
        return self

    def predict(self, x: np.typing.NDArray[np.floating]) -> Union[np.typing.NDArray[np.floating], float]:

        if not hasattr(self, "coef_"):
            raise ValueError(f"need to fit {self.__class__.__name__} first!")

        return x @ self.coef_


@dataclass
class ClusterExpansion:

    model: Model
    cluster_basis: ClusterBasis
    type_map: np.typing.NDArray[np.str_]

    def save(self, path: Path):

        warnings.warn(
            f"{self.__class__.__name__} uses pickle for now. This is unsecure! TODO write a serialization method"
        )

        with path.open("wb") as file:
            file.write(pickle.dumps(self))

    @classmethod
    def load(cls, path: Path) -> "ClusterExpansion":

        warnings.warn(
            f"{cls.__class__.__name__} uses pickle for now. This is unsecure! TODO write a serialization method"
        )

        with path.open("rb") as file:
            obj = pickle.load(file)

        if not isinstance(obj, cls):
            raise ValueError(f"loaded object is not of type {cls.__name__}")
        return obj


def train(
    configurations: list[Atoms],
    basis: ClusterBasis,
    model: Model = LimitingRidge(),
    target_property_computer: Optional[PropertyComputer] = None,
    feature_computer: Optional[FeatureComputer] = None,
) -> ClusterExpansion:

    """if not model:
        model: Model = LimitingRidge()"""

    if not target_property_computer:
        target_property_computer = total_energy

    type_map = get_type_map(configurations)
    if not feature_computer:
        feature_computer = topological_feature_vector_factory(basis=basis, type_map=type_map)

    model = model.fit(
        *get_data_pairs(
            configurations=configurations,
            basis=basis,
            target_property_computer=target_property_computer,
            feature_computer=feature_computer,
        )
    )

    return ClusterExpansion(model=model,cluster_basis=basis,type_map=type_map,)