from itertools import permutations

import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from ase.build import bulk
import sparse
import numpy as np

from tce.constants import STRUCTURE_TO_CUTOFF_LISTS, LatticeStructure
from tce import topology


def main():

    # Build simple cubic lattice
    atoms = bulk('Ar', crystalstructure='sc', a=1.0, cubic=True).repeat((2, 2, 2))

    # Find neighbors within cutoff distance (to define bonds)
    distances = atoms.get_all_distances(mic=True)

    tolerance = 0.01
    cutoffs = STRUCTURE_TO_CUTOFF_LISTS[LatticeStructure.SC][1:]
    adjacency_tensors = sparse.stack([
        sparse.DOK.from_numpy(
            np.logical_and((1.0 - tolerance) * cutoff < distances, distances < (1.0 + tolerance) * cutoff)
        ) for cutoff in cutoffs
    ])

    three_body_tensors = topology.get_three_body_tensors(
        lattice_structure=LatticeStructure.SC,
        adjacency_tensors=adjacency_tensors,
        max_three_body_order=2
    )

    # Set up 3D plot
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    verts = [
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1],
    ]

    # Define the 6 faces of the cube using the vertices indices
    faces = [
        [verts[0], verts[1], verts[2], verts[3]],  # bottom
        [verts[4], verts[5], verts[6], verts[7]],  # top
        [verts[0], verts[1], verts[5], verts[4]],  # front
        [verts[2], verts[3], verts[7], verts[6]],  # back
        [verts[1], verts[2], verts[6], verts[5]],  # right
        [verts[4], verts[7], verts[3], verts[0]],  # left
    ]

    # Create transparent cube
    cube = Poly3DCollection(faces, facecolors='lightblue', linewidths=1, edgecolors='black', linestyles="--", alpha=0.2)
    ax.add_collection3d(cube)

    # Plot atoms as blue spheres (dots)
    x, y, z = atoms.positions.T
    ax.scatter(x, y, z, color='black', s=500)

    # Plot bonds as black lines between neighbors
    for i, j in permutations(three_body_tensors[0].coords[:, 0], r=2):
        xs = [atoms.positions[i, 0], atoms.positions[j, 0]]
        ys = [atoms.positions[i, 1], atoms.positions[j, 1]]
        zs = [atoms.positions[i, 2], atoms.positions[j, 2]]
        ax.plot(xs, ys, zs, color='black', linewidth=3)

    for i, j in permutations(three_body_tensors[1].coords[:, 41], r=2):
        xs = [atoms.positions[i, 0], atoms.positions[j, 0]]
        ys = [atoms.positions[i, 1], atoms.positions[j, 1]]
        zs = [atoms.positions[i, 2], atoms.positions[j, 2]]
        ax.plot(xs, ys, zs, color='black', linewidth=3)

    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])

    ax.set_axis_off()
    plt.tight_layout()

    # Show plot
    plt.show()

    # To save as PNG uncomment the next line:
    # fig.savefig('simple_cubic_with_bonds.png', dpi=300)


if __name__ == "__main__":

    mpl.use("TkAgg")
    main()
