from __future__ import annotations

from typing import List, Dict, Any
import sparsene.op_gen.nvir.nvop as nvop

from sparsene.logging import get_logger

logger = get_logger(__name__)


class OpGraph:
    class Node:
        node_id: str
        inputs: List[OpGraph.Edge]
        outputs: List[OpGraph.Edge]
        op: nvop.NvOp

        def __init__(self, op: nvop.NvOp):
            self.node_id = op.name
            self.op = op
            self.inputs = []
            self.outputs = []

    class Edge:
        src: OpGraph.Node
        dst: OpGraph.Node

        def __init__(self, src: OpGraph.Node, dst: OpGraph.Node):
            self.src = src
            self.dst = dst

    nodes: Dict[str, Node]
    edges: List[Edge]

    def __init__(self):
        self.nodes = {}
        self.edges = []

    def add_node(self, node: Node):
        self.nodes[node.node_id] = node

    def is_directly_connected(self, src: str, dst: str) -> bool:
        for edge in self.nodes[src].outputs:
            if edge.dst.node_id == dst:
                return True
        return False

    def add_edge(self, edge: Edge):
        assert edge.src.node_id in self.nodes and edge.dst.node_id in self.nodes
        if self.is_directly_connected(edge.src.node_id, edge.dst.node_id):
            return

        self.edges.append(edge)
        self.nodes[edge.src.node_id].outputs.append(edge)
        self.nodes[edge.dst.node_id].inputs.append(edge)


def construct_graph_from_op_sequence(op_seq: nvop.NvOpSequence) -> OpGraph:
    graph = OpGraph()

    # Add nodes
    logger.debug(f"Adding OpNodes")
    for op in op_seq.ops:
        logger.debug(f"Adding node {op.name}: {op}")
        graph.add_node(OpGraph.Node(op))
    logger.debug(f"Nodes: {graph.nodes}")

    # Add edges
    for op in op_seq.ops:
        for inp in op.inputs:
            src = inp.tensor.source
            if not isinstance(src, nvop.NvOpOutput):
                # We only care about dependencies between ops
                continue
            src = src.op
            assert isinstance(src, nvop.NvOp)
            src_name = src.name
            dst_name = op.name
            logger.debug(f"src({src_name}): {src}")
            logger.debug(f"dst({dst_name}): {op}")
            logger.debug(
                f"Adding edge from {src_name} to {dst_name} "
                f"(Nodes exist: src={src_name in graph.nodes}, dst={dst_name in graph.nodes})"
            )
            if src_name in graph.nodes and dst_name in graph.nodes:
                logger.debug(f"Added edge from {src_name} to {dst_name}")
                graph.add_edge(
                    OpGraph.Edge(graph.nodes[src_name], graph.nodes[dst_name])
                )
    logger.debug(f"Edges: {graph.edges}")

    return graph
