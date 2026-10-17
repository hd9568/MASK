from sparsene.op_gen.nvir.opgraph.graph import OpGraph


class GraphPrinter:
    def __init__(self, indent_size: int = 3, debug_indent: bool = False):
        self.indent_size = indent_size
        self.debug_indent = debug_indent
        self.debug_chars = "."

    def _get_indent(self, indent_level: int) -> str:
        return (
            (self.debug_chars if self.debug_indent else " ")
            * indent_level
            * self.indent_size
        )

    def _indent_lines(self, text: str, indent_level: int = 1) -> str:
        """Helper method to indent all lines in a text string."""
        indent = self._get_indent(indent_level)
        return "\n".join(f"{indent}{line}" for line in text.split("\n"))

    def dump_node(self, node: OpGraph.Node, indent_level: int = 0) -> str:
        assert isinstance(node, OpGraph.OpNode)
        node_str = node.op.name
        return self._indent_lines(
            f"{node.node_id}: {node_str}", indent_level
        )

    def dump_graph(self, graph: OpGraph, indent_level: int = 0) -> str:
        nodes_str = (
            f"nodes={{\n"
            + self._indent_lines(
                "\n".join(self.dump_node(node) for node in graph.nodes.values()),
            )
            + "\n}"
        )
        edges_str = (
            f"edges={{\n"
            + self._indent_lines(
                "\n".join(
                    f"{edge.src.node_id} -> {edge.dst.node_id}"
                    for edge in graph.edges
                ),
            )
            + "\n}"
        )

        return self._indent_lines(
            f"OpGraph{{\n"
            f"{self._indent_lines(nodes_str)}\n"
            f"{self._indent_lines(edges_str)}\n"
            f"}}",
            indent_level,
        )

    def print_graph(self, graph: OpGraph):
        print(self.dump_graph(graph))
