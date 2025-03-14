from sparsene.op_gen.opgraph.graph import Graph
from sparsene.op_gen.opir import printer


class GraphPrinter:
    def __init__(self, graph: Graph):
        self.graph = graph

    def print_graph(self):
        for node in self.graph.nodes.values():
            print(node.id, printer.default_printer.dump(node.op))
        print("=" * 80)
        for edge in self.graph.edges:
            print(edge.src.id, edge.dst.id)
