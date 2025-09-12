from ase import build
from ase.calculators.eam import EAM
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from tce.training import train
from tce.constants import LatticeStructure, ClusterBasis


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
        configuration.symbols = generator.choice(
            a=species,
            p=[x_cu, 1.0 - x_cu],
            size=len(configuration)
        )

        configuration.calc = EAM(potential="Cu_Ni_Fischer_2018.eam.alloy")
        configurations.append(configuration)

    alpha_values = np.logspace(-4.0, 3.0, 20)
    prop_considered_clusters = np.zeros_like(alpha_values)

    for i, alpha in enumerate(alpha_values):

        pipeline = Pipeline([
            ("scale", StandardScaler()),
            ("reduce", PCA()),
            ("fit", Lasso(alpha=alpha))
        ])

        cluster_expansion = train(
            configurations=configurations,
            basis=ClusterBasis(
                lattice_structure=LatticeStructure.BCC,
                lattice_parameter=3.56,
                max_adjacency_order=3,
                max_triplet_order=2
            ),
            model=pipeline
        )

        prop_considered_clusters[i] = np.logical_not(
            np.isclose(pipeline["fit"].coef_, 0.0)
        ).mean()

    plt.plot(alpha_values, 100 * prop_considered_clusters, color="orchid")
    plt.xscale("log")
    plt.xlabel("regularization parameter")
    plt.ylabel("proportion of considered clusters (%)")
    plt.grid()
    plt.tight_layout()
    plt.savefig("regularization.png", dpi=800, bbox_inches="tight")


if __name__ == "__main__":

    main()
