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

from sparsene.op_gen.nvir.opgraph.graph import OpGraph, construct_graph_from_op_sequence

# NvOpProram{
#     meta_op = NvOpSequence{
#         gmem_tensor_op;
#         op0,
#         op1,
#         op2 = ForLoopOp{ # blk_y
#             body= NvOpSequence(
#                 op0, 
#                 op1, 
#                 ForLoopOp { # blk_x
#                     body = NvOpSequence(
#                         op0,
#                         op1,
#                         ForLoopOp { # intra-block for-loop
#                             # pipelined
#                             body= NvOpPipeline
#                         }
#                     )
#                 }
#             )
#         },
#         op3
#     }
# }

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


def validate_pipeline_plan(op_graph: OpGraph, pipeline_plan: PipelinePlan) -> bool:
    for shift in pipeline_plan.shifts:
        if shift < 1:
            # shift must be at least 1, or the two stages should be merged into one
            return False
    for stage in pipeline_plan.stages:
        for op_prior_idx in range(len(stage.ops)):
            for op_after_idx in range(op_prior_idx + 1, len(stage.ops)):
                op_prior = stage.ops[op_prior_idx]
                op_after = stage.ops[op_after_idx]
                if op_graph.is_directly_connected(op_after.name, op_prior.name):
                    # intra-stage back edge found
                    return False
    for stage_prior_idx in range(len(pipeline_plan.stages)):
        for stage_after_idx in range(stage_prior_idx + 1, len(pipeline_plan.stages)):
            stage_prior = pipeline_plan.stages[stage_prior_idx]
            stage_after = pipeline_plan.stages[stage_after_idx]
            for op_prior in stage_prior.ops:
                for op_after in stage_after.ops:
                    if op_graph.is_directly_connected(op_after.name, op_prior.name):
                        # inter-stage back edge found
                        return False
    return True


def enumerate_pipeline_plans(
    for_loop_op: ForLoopNvOp, max_shift: int = 3
) -> List[PipelinePlan]:
    assert not isinstance(
        for_loop_op.body, NvOpPipeline
    )  # The for loop should not be pipelined yet

    op_graph = construct_graph_from_op_sequence(for_loop_op.body)
