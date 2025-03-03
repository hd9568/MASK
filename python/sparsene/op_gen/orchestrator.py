from sparsene.op_gen.rts_parser import computent_from_rts, Computent
from sparsene.model.op import Op, SequentialOp, MetaOp
from sparsene.model.stage import Stage, Pipeline
from typing import List, Optional
from sparsene.op_gen.rts_parser import (
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
