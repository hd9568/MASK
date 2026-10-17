import argparse

from sparsene.op_gen.nvir.opgraph.graph import construct_graph_from_op_sequence as construct_graph_from_block_op
from sparsene.formats.DTC_SpMM import ME_TCF_FORMAT as DTC_SPMM_FORMAT
from sparsene.transform.rts import derive_rts
from sparsene.op_gen.computent.computent import computent_from_rts
from sparsene.op_gen.opir.generate import generate_from_computent
from sparsene.op_gen.opir.op_ir import ForLoopOp, MetaOp
from sparsene.op_gen.nvir.opgraph.printer import GraphPrinter
from sparsene.logging import get_logger, set_logging_level_for_all
from sparsene.op_gen.nvir.opgraph.graphviz_visualizer import visualize_op_graph as visualize_op_graph
from sparsene.op_gen.opir import printer
# from sparsene.op_gen.nvir.opgraph.visualizerx import visualize_op_graph as visualize_op_graphx

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
    print(printer.default_printer.dump(meta_op))
    assert isinstance(meta_op, MetaOp), f"meta_op: {meta_op}"
    dense_for_op = meta_op.block.ops[9]
    assert isinstance(dense_for_op, ForLoopOp), f"dense_for_op: {dense_for_op}"
    sparse_for_op = dense_for_op.block.ops[2]
    assert isinstance(sparse_for_op, ForLoopOp), f"sparse_for_op: {sparse_for_op}"

    printer = GraphPrinter()

    sparse_for_op_graph = construct_graph_from_block_op(sparse_for_op)
    printer.print_graph(sparse_for_op_graph)
    visualize_op_graph(sparse_for_op_graph, "sparse_for_op_graph.pdf")

    dense_for_op_graph = construct_graph_from_block_op(dense_for_op)
    printer.print_graph(dense_for_op_graph)
    visualize_op_graph(dense_for_op_graph, "dense_for_op_graph.pdf")

    whole_graph = construct_graph_from_block_op(meta_op)
    printer.print_graph(whole_graph)
    visualize_op_graph(whole_graph, "op_graph.pdf")

    # visualize_op_graphx(whole_graph, "op_graphx.pdf")