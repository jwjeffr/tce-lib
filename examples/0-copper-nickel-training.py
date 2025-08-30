from pathlib import Path

from ase import build
from ase.calculators.eam import EAM
import numpy as np
import requests

from tce.constants import LatticeStructure
from tce.training import ClusterBasis, LimitingRidge


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

        potential = "Cu_Ni_Fischer_2018.eam.alloy"
        if not Path(potential).exists():

            # from https://doi.org/10.1016/j.actamat.2019.06.027
            response = requests.get("https://www.ctcms.nist.gov/potentials/Download/2019--Fischer-F-Schmitz-G-Eich-S-M--Cu-Ni/3/Cu_Ni_Fischer_2018.eam.alloy")
            with open(potential, "w") as file:
                file.write(response.text)

        configuration.calc = EAM(potential=potential)
        configurations.append(configuration)

    model = LimitingRidge().fit(
        configurations,
        basis=ClusterBasis(
            lattice_structure=lattice_structure,
            lattice_parameter=lattice_parameter,
            max_adjacency_order=3,
            max_triplet_order=2
        )
    )

    model.save(Path("CuNi.npz"))


if __name__ == "__main__":

    main()
