from __future__ import annotations

from typing import List, Dict, Any
from sparsene.op_gen.opir import op_ir

from sparsene.logging import get_logger

logger = get_logger(__name__)


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

    class SuperNode:
        # TODO: Implement nested SuperNode
        graph: Graph

        def __init__(self, graph: Graph):
            self.graph = graph

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
        self.nodes[node.id] = node

    def is_directly_connected(self, src: int, dst: int) -> bool:
        for edge in self.nodes[src].outputs:
            if edge.dst.id == dst:
                return True
        return False

    def add_edge(self, edge: Edge):
        assert edge.src.id in self.nodes and edge.dst.id in self.nodes
        if self.is_directly_connected(edge.src.id, edge.dst.id):
            return

        self.edges.append(edge)
        self.nodes[edge.src.id].outputs.append(edge)
        self.nodes[edge.dst.id].inputs.append(edge)


def construct_graph_from_block(block: op_ir.Block) -> Graph:
    # TODO: Buggy. no edges are derived TnT
    graph = Graph()

    # Add nodes
    for op in block.ops:
        logger.debug(f"Adding node {id(op)}")
        graph.add_node(Graph.Node(op))
    logger.debug(f"Nodes: {graph.nodes}")

    # Add edges
    for op in block.ops:
        for operand in op.operands:
            src_id = id(operand.source.defining_op)
            dst_id = id(op)
            logger.debug(f"src({src_id}): {operand.source.defining_op}")
            logger.debug(f"dst({dst_id}): {op}")
            logger.debug(f"Adding edge from {src_id} to {dst_id}")
            logger.debug(f"src_id in graph.nodes: {src_id in graph.nodes}")
            logger.debug(f"dst_id in graph.nodes: {dst_id in graph.nodes}")
            if src_id in graph.nodes and dst_id in graph.nodes:
                logger.debug(f"Added")
                graph.add_edge(Graph.Edge(graph.nodes[src_id], graph.nodes[dst_id]))

    return graph


# class OpGraph:
#     class Node:
#         pass

#     class Edge:
#         pass

#     nodes: Dict[int, Node]
#     edges: List[Edge]
