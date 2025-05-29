from itertools import combinations_with_replacement

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import ovito

from cowley_sro_parameters import nearest_neighbor_topology, sro_modifier


# type map constant info
TYPE_MAP = {1: 'Co', 2: 'Ni', 3: 'Cr', 4: 'Fe', 5: 'Mn'}
INVERSE_TYPE_MAP = {val: key for key, val in TYPE_MAP.items()}
NUM_TYPES = len(TYPE_MAP)

# cutoff info for SRO calculations
NUM_NEAREST_NEIGHBORS = 12


def main():

    # create ovito pipeline, add a bonds modifier to create needed topology
    pipeline = ovito.io.import_file("sim/cluster_expansion-*.dat")
    nearest_neighbor_modifier = nearest_neighbor_topology(NUM_NEAREST_NEIGHBORS)
    pipeline.modifiers.append(nearest_neighbor_modifier)

    # create SRO modifier which calculates all SRO's
    modifier = sro_modifier(type_map=TYPE_MAP)
    pipeline.modifiers.append(modifier)

    pairs = list(combinations_with_replacement(TYPE_MAP.values(), 2))
    sro_params = np.zeros((pipeline.source.num_frames, NUM_TYPES, NUM_TYPES))
    for frame in range(pipeline.source.num_frames):

        data = pipeline.compute(frame)
        for e1, e2 in pairs:
            i, j = INVERSE_TYPE_MAP[e1], INVERSE_TYPE_MAP[e2]
            sro_params[frame, i - 1, j - 1] = data.attributes[f'sro_{e1}{e2}']

    fig = plt.figure()
    gs = GridSpec(NUM_TYPES, NUM_TYPES, figure=fig, wspace=0.3, hspace=0.3)

    for i in range(NUM_TYPES):
        for j in range(i + 1):
            t = TYPE_MAP[i + 1], TYPE_MAP[j + 1]
            ax = plt.subplot(gs[i, j])
            ax.set_ylim([-1.1, 1.1])
            ax.set_xticks([0, 20, 40, 60, 80, 100])
            ax.set_yticks([-1, -0.5, 0, 0.5, 1])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.grid()
            # along the left side of the grid, label y-axis ticks
            if j == 0:
                ax.set_yticklabels([-1, '', 0, '', 1])

                # add y-axes label in the middle
                if i == (NUM_TYPES + 1) // 2 - 1:
                    ax.set_ylabel(r"$\alpha$-$\alpha'$ Cowley SRO parameter ($\chi_{\alpha\alpha'}$)")

            # along the bottom side of the grid, label x-axis ticks
            if i == NUM_TYPES - 1:
                ax.set_xticklabels([0, '', '', '', '', 100])

                # add x-axes label in the middle
                if j == (NUM_TYPES + 1) // 2 - 1:
                    ax.set_xlabel('frame')

            # along the diagonal, label atom type names
            if i == j:
                ax.text(50, 1.3, r"$\alpha = $" + t[1], ha='center', va='bottom')
                ax.text(110, 0.0, r"$\alpha' = $" + t[0], ha='left', va='center')
            ax.plot(sro_params[:, j, i], color='black')

    fig.tight_layout()
    fig.savefig("cluster-expansion-mc.pdf", bbox_inches="tight")


if __name__ == '__main__':

    mpl.use('TkAgg')
    main()
