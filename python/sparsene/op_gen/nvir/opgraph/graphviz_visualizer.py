import graphviz
from .graph import OpGraph


def generate_dot_file(graph: OpGraph, filename: str):
    dot = graphviz.Digraph(comment="OpGraph")

    for node in graph.nodes.values():
        dot.node(node.node_id, f"{node.node_id}")

    for edge in graph.edges:
        dot.edge(edge.src.node_id, edge.dst.node_id)

    dot.render(filename, format="png")


def visualize_op_graph(graph: OpGraph, filename: str):
    generate_dot_file(graph, filename)
