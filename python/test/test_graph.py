import argparse

from sparsene.op_gen.opgraph.graph import construct_graph_from_block_op, OpGraph
from sparsene.formats.DTC_SpMM import DTC_SPMM_FORMAT
from sparsene.transform.rts import derive_rts
from sparsene.op_gen.computent.computent import computent_from_rts
from sparsene.op_gen.opir.generate import generate_from_computent
from sparsene.op_gen.opir.op_ir import ForLoopOp, MetaOp
from sparsene.op_gen.opgraph.printer import GraphPrinter
from sparsene.logging import get_logger, set_logging_level_for_all


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--debug", "-d", action="store_true")
    return args.parse_args()


if __name__ == "__main__":
    args = parse_args()
    set_logging_level_for_all("DEBUG" if args.debug else "INFO")

    logger = get_logger("test")
    rts = derive_rts(DTC_SPMM_FORMAT)
    computent = computent_from_rts("dtc_spmm", rts)
    meta_op = generate_from_computent(computent)
    assert isinstance(meta_op, MetaOp), f"meta_op: {meta_op}"
    dense_for_op = meta_op.block.ops[2]
    assert isinstance(dense_for_op, ForLoopOp), f"dense_for_op: {dense_for_op}"
    sparse_for_op = dense_for_op.block.ops[3]
    assert isinstance(sparse_for_op, ForLoopOp), f"sparse_for_op: {sparse_for_op}"

    printer = GraphPrinter()

    graph = construct_graph_from_block_op(sparse_for_op)
    printer.print_graph(graph)

    whole_graph = construct_graph_from_block_op(meta_op)
    printer.print_graph(whole_graph)
