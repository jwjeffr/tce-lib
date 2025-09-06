r"""
This module defines some training convenience wrappers for easily training and serializing a cluster expansion model
from a list of configurations, encoded as `ase.Atoms` objects.
"""

from dataclasses import dataclass
from abc import abstractmethod
from typing import Callable, TypeAlias, Union, Optional, Protocol, runtime_checkable
import warnings
from pathlib import Path
import pickle

import numpy as np
from ase import Atoms

from tce.constants import ClusterBasis, STRUCTURE_TO_ATOMIC_BASIS
from tce.topology import FeatureComputer, topological_feature_vector_factory
from tce import __url__


NON_CUBIC_CELL_MESSAGE = "At least one of your configurations has a non-cubic cell. For now, tce-lib does not support non-cubic lattices."
"""@private"""

INCOMPATIBLE_GEOMETRY_MESSAGE = "Geometry in all configurations must match geometry in cluster basis."
"""@private"""

NO_POTENTIAL_ENERGY_MESSAGE = "At least one of your configurations does not have a computable potential energy."
"""@private"""

LARGE_SYSTEM_THRESHOLD = 1_000
"""@private"""

LARGE_SYSTEM_MESSAGE = f"You have passed a relatively large system (larger than {LARGE_SYSTEM_THRESHOLD:.0f}) as a training point. This will be very slow."
"""@private"""


def get_type_map(configurations: list[Atoms]) -> np.typing.NDArray[np.str_]:

    r"""
    function that generates a species ordering for a list of configurations. this grabs all chemical types available
    within the list of configurations, and then sorts them lexographically

    Args:
        configurations (list[Atoms]):
            list of atomic configurations
    """

    # not all configurations need to have the same number of types, calculate the union of types
    all_types = set.union(*(set(x.get_chemical_symbols()) for x in configurations))
    return np.array(sorted(list(all_types)))


PropertyComputer: TypeAlias = Callable[[Atoms], Union[float, np.typing.NDArray[np.floating]]]
r"""
Type alias for computing a property from an `ase.Atoms` object. In general, this will be a function mapping a 
configuration (an `ase.Atoms` object) to a target property, which should be either a float or an array. For example, 
for computing the total energy:

```py
from ase import Atoms

def total_energy(atoms: Atoms) -> float:
    return atoms.get_potential_energy()
```
"""

def total_energy(atoms: Atoms) -> float:

    r"""
    Property computer which computes the total energy, and raises an error if the energy is not available.

    Args:
        atoms (Atoms):
            configuration to compute the energy for
    """

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

    r"""
    Function to create datapairs, i.e. a sequence of features and target properties $(\mathbf{X}, y)$.

    Args:
        configurations (list[Atoms]):
            list of atomic configurations to compute features and target properties from
        basis (ClusterBasis):
            cluster basis of which to compute features from
        target_property_computer (PropertyComputer):
            property computer that computes target properties. If not specified, the target property is set to total
            energy
        feature_computer (FeatureComputer):
            feature computer that computes features. If not specified, the feature is set to the topological feature
            vector computed by `tce.topology.topological_feature_vector_factory`.
    """

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

    r"""
    Model protocol which defines the contract of how a model should behave. This closely follows the `scikit-learn`
    contract, i.e. an object with a `.fit` method and a `.predict` method.
    """

    @abstractmethod
    def fit(self, X: np.typing.NDArray[np.floating], y: np.typing.NDArray[np.floating]) -> "Model":

        r"""
        fit with a data matrix $X$ and a target matrix $y$

        Args:
            X (np.typing.NDArray[np.floating]):
                data matrix
            y (np.typing.NDArray[np.floating]):
                target matrix
        """

        pass

    @abstractmethod
    def predict(self, x: np.typing.NDArray[np.floating]) -> Union[np.typing.NDArray[np.floating], float]:

        r"""
        predict for a particular data vector $x$

        Args:
            x (np.typing.NDArray[np.floating]):
                data vector
        """

        pass


class LimitingRidge:

    r"""
    train by minimizing the limiting ridge loss:

    $$L(\beta \; | \; \lambda) = \|X\beta - y\|_2^2 + \lambda \|\beta\|_2^2$$

    $$\hat{\beta} = \lim_{\lambda\to 0^+} L(\beta\;|\;\lambda) $$
    """

    def fit(self, X: np.typing.NDArray[np.floating], y: np.typing.NDArray[np.floating]) -> "Model":

        r"""
        fit using the Moore penrose inverse, i.e. $\hat{\beta} = X^+y$, and store the coefficients

        Args:
            X (np.typing.NDArray[np.floating]):
                data matrix
            y (np.typing.NDArray[np.floating]):
                target matrix
        """

        self.coef_ = np.linalg.pinv(X) @ y
        return self

    def predict(self, x: np.typing.NDArray[np.floating]) -> Union[np.typing.NDArray[np.floating], float]:

        r"""
        predict for a particular data vector $x$, i.e. $\hat{y} = x^\intercal \hat{\beta}

        Args:
            x (np.typing.NDArray[np.floating]):
                data vector
        """

        if not hasattr(self, "coef_"):
            raise ValueError(f"need to fit {self.__class__.__name__} first!")

        return x @ self.coef_


@dataclass
class ClusterExpansion:

    fr"""
    Cluster expansion data container, mostly useful for training a model, saving a model, and deploying it elsewhere,
    like in the example [here]({__url__}#training-monte-carlo)
    """

    model: Model
    cluster_basis: ClusterBasis
    type_map: np.typing.NDArray[np.str_]

    def save(self, path: Path):

        r"""
        method to save the model using the pickle library. **WARNING**, this is not a secure method! Only load from a
        source that you really trust.

        Args:
            path (Path):
                path to save the model to
        """

        warnings.warn(
            f"{self.__class__.__name__} uses pickle for now. This is unsecure! TODO write a serialization method"
        )

        with path.open("wb") as file:
            file.write(pickle.dumps(self))

    @classmethod
    def load(cls, path: Path) -> "ClusterExpansion":
        r"""
        method to load a serialized model using the pickle library. **WARNING**, this is not a secure method! Only load
        from a source that you really trust.

        Args:
            path (Path):
                path to load the model from
        """

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

    r"""
    Convenience training method wrapper. Here, we train on a list of configurations and output a cluster expansion
    model.

    Args:
        configurations (list[Atoms]):
            list of configurations to train on
        basis (ClusterBasis):
            cluster basis
        model (Model, optional):
            model used to train the model. If not specified, defaults to `training.LimitingRidge`.
        target_property_computer (PropertyComputer, optional):
            target property computer to use when training the model. If not specified, defaults to computing the total
            energy.
        feature_computer (FeatureComputer, optional):
            feature computer to use when training the model. If not specified, defaults to computing the topological
            feature vector.
    """

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