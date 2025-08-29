import ovito
from cowley_sro_parameters import nearest_neighbor_topology, sro_modifier
import matplotlib.pyplot as plt


def main():

    pipeline = ovito.io.import_file("copper-nickel/frame_*.xyz")
    pipeline.modifiers.append(nearest_neighbor_topology(num_neighbors=8))
    pipeline.modifiers.append(sro_modifier())

    sro_parameters = [data.attributes["sro_12"] for data in pipeline.frames]
    plt.plot(sro_parameters, color="mediumvioletred")
    plt.grid()
    plt.xlabel("Monte Carlo frame")
    plt.ylabel("Cu-Ni Cowley short range order parameter")
    plt.tight_layout()
    plt.savefig("cu-ni-sro.png", dpi=800, bbox_inches="tight")


if __name__ == "__main__":

    main()
