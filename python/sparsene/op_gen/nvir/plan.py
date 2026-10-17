from sparsene.op_gen.nvir.nvop import (
    NvOpPipeline,
    NvOpSequence,
    ForLoopNvOp,
)
from typing import List, Sequence


class PipelinePlan:
    stages: List[NvOpSequence]
    shifts: List[int]

    def __init__(self, stages: Sequence[NvOpSequence], shifts: Sequence[int]):
        assert len(stages) > 0, "Pipeline plan must have at least one stage"
        assert (
            len(stages) == len(shifts) + 1
        ), "Number of stages must be one more than number of shifts"
        self.stages = list(stages)
        self.shifts = list(shifts)

    def __str__(self) -> str:
        stages = [
            f"{', '.join([op.name for op in stage.ops])} {f'|({self.shifts[i]})>' if i < len(self.shifts) else ''}"
            for i, stage in enumerate(self.stages)
        ]
        return " ".join(stages)


def apply_pipeline(for_loop_op: ForLoopNvOp, pipeline_plan: PipelinePlan) -> None:
    pipeline = NvOpPipeline(
        name=for_loop_op.body.name,
        stages=pipeline_plan.stages,
        shifts=pipeline_plan.shifts,
    )
    for op in pipeline.ops:
        op.pipelined = True
    for_loop_op.body = pipeline
