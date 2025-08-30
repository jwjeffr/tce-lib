from dataclasses import dataclass

from ase import Atoms, build
from ase.calculators.eam import EAM
import numpy as np
from sklearn.linear_model._base import LinearModel
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from tce.training import TrainingMethod, ClusterBasis, CEModel, get_data_pairs
from tce.constants import LatticeStructure


@dataclass
class SKLearnLinearRegression(TrainingMethod):

    model: LinearModel

    def fit(self, configurations: list[Atoms], basis: ClusterBasis) -> CEModel:

        # not all configurations need to have the same number of types, calculate the union of types
        all_types = set.union(*(set(x.get_chemical_symbols()) for x in configurations))
        type_map = np.array(sorted(list(all_types)))

        X, y = get_data_pairs(configurations, basis)
        X = StandardScaler().fit_transform(X)
        y = StandardScaler().fit_transform(y.reshape(-1, 1)).squeeze()

        return CEModel(
            cluster_basis=basis,
            interaction_vector=self.model.fit(X, y).coef_,
            type_map=type_map
        )


def main():

    lattice_parameter = 3.56
    lattice_structure = LatticeStructure.BCC
    species = np.array(["Cu", "Ni"])
    generator = np.random.default_rng(seed=0)

    atoms = build.bulk(
        name=species[0],
        crystalstructure=lattice_structure.name.lower(),
        a=lattice_parameter,
        cubic=True
    ).repeat((3, 3, 3))

    num_configurations = 50
    configurations = []
    for _ in range(num_configurations):
        configuration = atoms.copy()
        x_cu = generator.random()
        configuration.symbols = generator.choice(a=species, p=[x_cu, 1.0 - x_cu], size=len(configuration))

        configuration.calc = EAM(potential="Cu_Ni_Fischer_2018.eam.alloy")
        configurations.append(configuration)

    alpha_values = np.logspace(-6.0, 2.0, 20)
    prop_considered_clusters = np.zeros_like(alpha_values)

    for i, alpha in enumerate(alpha_values):

        model = SKLearnLinearRegression(
            model=Lasso(alpha=alpha, fit_intercept=False, random_state=0, max_iter=1_000_000, warm_start=True)
        ).fit(
            configurations=configurations,
            basis=ClusterBasis(
                lattice_structure=LatticeStructure.BCC,
                lattice_parameter=3.56,
                max_adjacency_order=3,
                max_triplet_order=2
            )
        )

        prop_considered_clusters[i] = np.logical_not(np.isclose(model.interaction_vector, 0.0)).mean()

    plt.plot(alpha_values, 100 * prop_considered_clusters, color="orchid")
    plt.xscale("log")
    plt.xlabel("regularization parameter")
    plt.ylabel("proportion of considered clusters (%)")
    plt.grid()
    plt.tight_layout()
    plt.savefig("regularization.png", dpi=800, bbox_inches="tight")


if __name__ == "__main__":

    main()
