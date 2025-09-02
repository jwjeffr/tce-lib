import random

import networkx as nx
import matplotlib.pyplot as plt
from PIL import Image


TIGER_ORANGE = "#c88141"
REGALIA = "#522d80"


def main():
    pixels = 1000
    dpi = 8000

    figsize_in = pixels / dpi
    plt.figure(figsize=(figsize_in, figsize_in), dpi=dpi)

    G = nx.Graph()

    # Add two nodes
    G.add_node(1, color=TIGER_ORANGE)
    G.add_node(2, color=REGALIA)
    G.add_node(3, color=TIGER_ORANGE)

    # Optionally, add an edge
    G.add_edge(1, 2)
    G.add_edge(2, 3)
    G.add_edge(3, 1)

    # Extract positions
    # pos = nx.spring_layout(G, k=5.2)
    pos = nx.circular_layout(G)

    # Extract colors
    colors = [G.nodes[n]['color'] for n in G.nodes]

    nx.draw_networkx_edges(G, pos=pos, edge_color="black", width=0.1)
    nx.draw_networkx_nodes(G, pos=pos, node_color=colors, node_size=5, edgecolors="black", linewidths=0.1)

    plt.axis("equal")
    plt.axis("off")
    plt.margins(0.3)
    plt.savefig("favicon.png", dpi=dpi, pad_inches=0, transparent=True)
    plt.close()

    img = Image.open("favicon.png")
    width, height = img.size
    p = 0.1
    trim_x = int(width * p)
    trim_y = int(height * p)
    img.crop((trim_x, trim_y, width - trim_x, height - trim_y)).save("favicon.png", transparent=True)


if __name__ == "__main__":

    random.seed(0)
    main()