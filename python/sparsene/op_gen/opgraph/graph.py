from __future__ import annotations

from typing import List, Dict, Any
from sparsene.op_gen.opir import op_ir


class Graph:
    class Node:
        id: int
        op: op_ir.Op
        inputs: List[Graph.Edge]
        outputs: List[Graph.Edge]

        def __init__(self, op: op_ir.Op):
            self.id = id(op)
            self.op = op
            self.inputs = []
            self.outputs = []

    class Edge:
        src: Graph.Node
        dst: Graph.Node

        def __init__(self, src: Graph.Node, dst: Graph.Node):
            self.src = src
            self.dst = dst

        nodes: Dict[int, Graph.Node]
        edges: List[Graph.Edge]

    def __init__(self):
        self.nodes = {}
        self.edges = []

    def add_node(self, node: Node):
        self.nodes[id(node)] = node

    def is_directly_connected(self, src: int, dst: int) -> bool:
        for edge in self.nodes[src].outputs:
            if edge.dst.uid == dst:
                return True
        return False

    def add_edge(self, edge: Edge):
        assert edge.src.id in self.nodes and edge.dst.id in self.nodes
        if self.is_directly_connected(edge.src.id, edge.dst.id):
            raise ValueError(
                f"Node {edge.src.id} is already connected to {edge.dst.id}"
            )

        self.edges.append(edge)
        self.nodes[edge.src.id].outputs.append(edge)
        self.nodes[edge.dst.id].inputs.append(edge)


def construct_graph_from_block(block: op_ir.Block) -> Graph:
    # TODO: Buggy. no edges are derived TnT
    graph = Graph()

    # Add nodes
    for op in block.ops:
        graph.add_node(Graph.Node(op))

    # Add edges
    for op in block.ops:
        for operand in op.operands:
            src_id = id(operand.source)
            dst_id = id(op)
            if src_id in graph.nodes and dst_id in graph.nodes:
                graph.add_edge(Graph.Edge(graph.nodes[src_id], graph.nodes[dst_id]))

    return graph


# class OpGraph:
#     class Node:
#         pass

#     class Edge:
#         pass

#     nodes: Dict[int, Node]
#     edges: List[Edge]
