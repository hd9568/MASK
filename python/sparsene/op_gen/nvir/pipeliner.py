from sparsene.op_gen.nvir.nvop import (
    NvOpPipeline,
    NvOpProgram,
    NvOpSequence,
    NvOpTensor,
    NvOpInput,
    NvOpOutput,
    NvOp,
    ForLoopNvOp,
)
from dataclasses import dataclass
from typing import List


@dataclass
class PipelinePlan:
    stages: List[NvOpSequence]
    shifts: List[int]

    def __post_init__(self):
        assert len(self.stages) == len(self.shifts) + 1


def apply_pipeline(for_loop_op: ForLoopNvOp, pipeline_plan: PipelinePlan) -> None:
    pipeline = NvOpPipeline(
        name=for_loop_op.body.name,
        stages=pipeline_plan.stages,
        shifts=pipeline_plan.shifts,
    )
    for op in pipeline.ops:
        op.pipelined = True
    for_loop_op.body = pipeline
