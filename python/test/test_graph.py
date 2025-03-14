import argparse

from sparsene.op_gen.opgraph.graph import construct_graph_from_block, Graph
from sparsene.formats.DTC_SpMM import DTC_SPMM_FORMAT
from sparsene.transform.rts import derive_rts
from sparsene.op_gen.computent.computent import computent_from_rts
from sparsene.op_gen.opir.generate import generate_from_computent
from sparsene.op_gen.opir.op_ir import ForLoopOp
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
    ops = generate_from_computent(computent)
    assert isinstance(ops[0], ForLoopOp)
    assert isinstance(ops[0].body.ops[3], ForLoopOp)
    graph = construct_graph_from_block(ops[0].body.ops[3].body)
    printer = GraphPrinter(graph)
    printer.print_graph()
