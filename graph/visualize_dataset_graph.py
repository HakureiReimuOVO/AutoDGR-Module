import argparse
import networkx as nx
import matplotlib.pyplot as plt
import pickle as pkl


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize dataset graph")
    parser.add_argument("graph", help="Graph file to visualize")
    args = parser.parse_args()
    return args


def load_graph(graph_file):
    with open(graph_file, "rb") as f:
        G = pkl.load(f)
    return G


def visualize_graph(G):
    pos = nx.spring_layout(G)
    plt.figure(figsize=(12, 12))

    nx.draw_networkx_nodes(G, pos, node_size=500, node_color="skyblue")

    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)

    labels = {
        node: f"{data['name']}\n{data['feature'].shape}"
        for node, data in G.nodes(data=True)
    }
    nx.draw_networkx_labels(G, pos, labels, font_size=8)

    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    plt.title("Dataset Graph")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    args = parse_args()
    G = load_graph(args.graph)
    visualize_graph(G)
