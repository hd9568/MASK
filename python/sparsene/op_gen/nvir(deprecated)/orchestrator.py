from sparsene.op_gen.computent.computent import computent_from_rts, Computent
from .op import Op, SequentialOp, MetaOp
from .stage import Stage, Pipeline
from typing import List, Optional
from sparsene.op_gen.computent.computent import (
    Schedule,
    DenseAxisIterator,
    SparseAxisIterator,
    SparseAxisSlicer,
    AtomicFormatOp,
    AtomicFormatType,
)


def orchestrate_pipeline(op_list: List) -> Pipeline:
    # TODO: IMPLEMENT
    return Pipeline(
        name="",
        stages=[],
        shifts=[],
    )
