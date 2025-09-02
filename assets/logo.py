from typing import Callable
from itertools import pairwise
from functools import wraps, partial

from ase import Atoms
import numpy as np
from ovito.io.ase import ase_to_ovito
from ovito.data import DataCollection
from ovito.vis import Viewport, TachyonRenderer
from ovito.pipeline import StaticSource, Pipeline

from tce.structures import Supercell
from tce.constants import LatticeStructure


COLORS = {
    1: (0.97, 0.40, 0.08),
    2: (0.32, 0.18, 0.50)
}


def type_map_modifier(frame: int, data: DataCollection, color_map: dict) -> None:

    # get integer labels of types
    types = data.particles_.particle_types_

    # assign atom type name and radius to types
    for key, color in color_map.items():
        types.type_by_id_(key).color = color

def smaller_radius_modifier(frame: int, data: DataCollection) -> None:

    # get integer labels of types
    types = data.particles_.particle_types_
    for unique_type in np.unique(types[...]):
        types.type_by_id_(unique_type).radius = 0.75 * types.type_by_id_(unique_type).radius


def make_bonds_modifier(groups: list[list[int]]) -> Callable[[int, DataCollection], None]:

    @wraps(make_bonds_modifier)
    def wrapper(frame: int, data: DataCollection) -> None:

        topology = set()
        for group in groups:
            for pair in pairwise(group):
                topology.add(pair)
            topology.add((group[-1], group[0]))
        bonds = data.particles_.create_bonds(count=len(topology))
        bonds.create_property('Topology', data=list(topology))

        transparency = np.zeros(len(data.particles_.positions))
        flattened_groups = set(sum(groups, []))

        for i in range(len(data.particles_.positions)):
            if i in flattened_groups:
                transparency[i] = 0.0
            else:
                transparency[i] = 0.75
        data.particles_.create_property("Transparency", data=transparency)

    return wrapper


def main():

    rng = np.random.default_rng(seed=0)

    supercell = Supercell(
        lattice_parameter=3.0,
        lattice_structure=LatticeStructure.SC,
        size=(3, 5, 3)
    )

    symbols = rng.choice(["Fe", "Cr"], size=supercell.num_sites)
    atoms = Atoms(
        symbols=symbols,
        pbc=True,
        positions=supercell.positions,
        cell=supercell.lattice_parameter * np.array(supercell.size)
    )

    groups = [
        [35, 20],
        [32, 31],
        [0, 15],
        [44, 29, 26],
        [12, 13, 9],
        [3, 4, 7]
    ]

    source = StaticSource(data=ase_to_ovito(atoms))
    source.data.cell.vis.enabled = False
    pipeline = Pipeline(source=source)
    pipeline.modifiers.append(smaller_radius_modifier)
    pipeline.modifiers.append(partial(type_map_modifier, color_map=COLORS))
    pipeline.modifiers.append(make_bonds_modifier(groups=groups))
    pipeline.add_to_scene()

    viewport = Viewport(type=Viewport.Type.Perspective, camera_dir=(2, 1, -1))
    size = (4000, 3000)
    viewport.zoom_all(size)
    viewport.render_image(filename="logo.png", size=size, renderer=TachyonRenderer(), alpha=True)


if __name__ == "__main__":

    main()