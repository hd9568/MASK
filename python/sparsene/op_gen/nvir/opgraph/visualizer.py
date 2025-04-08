import networkx as nx
import matplotlib.pyplot as plt
from sparsene.op_gen.nvir.opgraph.graph import OpGraph


def visualize_op_graph(graph: OpGraph, filename: str) -> None:
    """
    Visualize an OpGraph using networkx and matplotlib.

    Args:
        graph: The OpGraph instance to visualize
    """
    # Create a new directed graph
    G = nx.DiGraph()

    # Add nodes with labels
    for node_id, node in graph.nodes.items():
        if isinstance(node, OpGraph.OpNode):
            label = f"Op\n{node.op}"
            color = "lightgreen"
        else:
            raise ValueError(f"Unsupported node type: {type(node)}")

        G.add_node(node_id, label=label, color=color)

    # Add edges
    for edge in graph.edges:
        G.add_edge(edge.src.node_id, edge.dst.node_id)

    # Create the layout
    pos = nx.spring_layout(G, k=1, iterations=50)

    # Draw the graph
    plt.figure(figsize=(12, 8))

    # Draw nodes
    node_colors = [G.nodes[node]["color"] for node in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=2000)

    # Draw edges
    nx.draw_networkx_edges(G, pos, edge_color="gray", arrows=True, arrowsize=20)

    # Add labels
    labels = nx.get_node_attributes(G, "label")
    nx.draw_networkx_labels(G, pos, labels, font_size=8)

    plt.title("Operation Graph Visualization")
    plt.axis("off")
    plt.tight_layout()

    # Save the plot to a pdf
    plt.savefig(filename)
