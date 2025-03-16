from __future__ import annotations

from typing import List, Dict, Any
from sparsene.op_gen.opir import op_ir

from sparsene.logging import get_logger

logger = get_logger(__name__)


class OpGraph:
    class Node:
        id: int
        inputs: List[OpGraph.Edge]
        outputs: List[OpGraph.Edge]

        def __init__(self, id: int):
            self.id = id
            self.inputs = []
            self.outputs = []

    class BlockArgNode(Node):
        block_arg: op_ir.BlockArgument

        def __init__(self, block_arg: op_ir.BlockArgument):
            super().__init__(id(block_arg))
            self.block_arg = block_arg

    class OpNode(Node):
        op: op_ir.Op

        def __init__(self, op: op_ir.Op):
            super().__init__(id(op))
            self.op = op

    class SuperNode(OpNode):
        graph: OpGraph

        def __init__(self, op: op_ir.Op, graph: OpGraph):
            super().__init__(op)
            self.graph = graph

    class Edge:
        src: OpGraph.Node
        dst: OpGraph.Node

        def __init__(self, src: OpGraph.Node, dst: OpGraph.Node):
            self.src = src
            self.dst = dst

    nodes: Dict[int, Node]
    edges: List[Edge]

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


def construct_graph_from_block_op(block_op: op_ir.BlockOp) -> OpGraph:
    graph = OpGraph()

    # Add nodes
    logger.debug(f"Adding OpNodes")
    for op in block_op.block.ops:
        logger.debug(f"Adding node {id(op)}: {op}")
        if isinstance(op, op_ir.BlockOp):
            subgraph = construct_graph_from_block_op(op)
            graph.add_node(OpGraph.SuperNode(op, subgraph))
        else:
            graph.add_node(OpGraph.OpNode(op))

    logger.debug(f"Adding BlockArgument nodes")
    for arg in block_op.block.args:
        logger.debug(f"Adding node {id(arg)}: {arg}")
        graph.add_node(OpGraph.BlockArgNode(arg))

    logger.debug(f"Nodes: {graph.nodes}")

    # Add edges
    for op in block_op.block.ops:
        for operand in op.operands:
            src_value = operand.source
            if isinstance(src_value, op_ir.BlockArgument):
                src = src_value
            elif isinstance(src_value, op_ir.OpResult):
                src = src_value.defining_op
            else:
                raise ValueError(f"Operand type {type(src_value)} not supported")

            src_id = id(src)
            dst_id = id(op)
            logger.debug(f"src({src_id}): {src}")
            logger.debug(f"dst({dst_id}): {op}")
            logger.debug(
                f"Adding edge from {src_id} to {dst_id} "
                f"(Nodes exist: src={src_id in graph.nodes}, dst={dst_id in graph.nodes})"
            )
            if src_id in graph.nodes and dst_id in graph.nodes:
                logger.debug(f"Added edge from {src_id} to {dst_id}")
                graph.add_edge(OpGraph.Edge(graph.nodes[src_id], graph.nodes[dst_id]))
    logger.debug(f"Edges: {graph.edges}")

    return graph
