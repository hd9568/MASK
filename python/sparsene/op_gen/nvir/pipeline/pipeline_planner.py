import abc
import time
from sparsene.op_gen.nvir.opgraph.graph import OpGraph, construct_graph_from_op_sequence
from itertools import product
from typing import Dict
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
from sparsene.op_gen.nvir.plan import PipelinePlan
from typing import List
from sparsene.op_gen.nvir.pipeline.int_planner import (
    generate_all_partitions_with_constraints,
)


class PlanValidator(abc.ABC):
    @abc.abstractmethod
    def validate(self, pipeline_plan: PipelinePlan) -> bool: ...


class BasicValidator(PlanValidator):
    def __init__(self, op_graph: OpGraph):
        self.op_graph = op_graph

    def validate(self, pipeline_plan: PipelinePlan) -> bool:
        op_graph = self.op_graph
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
            for stage_after_idx in range(
                stage_prior_idx + 1, len(pipeline_plan.stages)
            ):
                stage_prior = pipeline_plan.stages[stage_prior_idx]
                stage_after = pipeline_plan.stages[stage_after_idx]
                for op_prior in stage_prior.ops:
                    for op_after in stage_after.ops:
                        if op_graph.is_directly_connected(op_after.name, op_prior.name):
                            # inter-stage back edge found
                            return False
        return True


class NeighborDependencyValidator(PlanValidator):
    def __init__(self, op_graph: OpGraph):
        self.op_graph = op_graph

    def validate(self, pipeline_plan: PipelinePlan) -> bool:
        op_graph = self.op_graph
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
            for stage_after_idx in range(
                stage_prior_idx + 1, len(pipeline_plan.stages)
            ):
                stage_prior = pipeline_plan.stages[stage_prior_idx]
                stage_after = pipeline_plan.stages[stage_after_idx]
                for op_prior in stage_prior.ops:
                    for op_after in stage_after.ops:
                        if op_graph.is_directly_connected(op_after.name, op_prior.name):
                            # inter-stage back edge found
                            return False
                        if (
                            op_graph.is_directly_connected(op_prior.name, op_after.name)
                            and stage_after_idx - stage_prior_idx > 1
                        ):
                            # inter-stage distant dependency found
                            return False
        return True


def enumerate_pipeline_plans(
    for_loop_op: ForLoopNvOp,
    validator: PlanValidator,
    min_nstages: int = 1,
    max_nstages: int = 99,
    min_ops_per_stage: int = 1,
    max_ops_per_stage: int = 99,
    min_shift: int = 1,
    max_shift: int = 3,
) -> List[PipelinePlan]:
    start_time = time.time()
    assert not isinstance(
        for_loop_op.body, NvOpPipeline
    )  # The for loop should not be pipelined yet

    op_graph = construct_graph_from_op_sequence(for_loop_op.body)

    op_name2idx: Dict[str, int] = {}
    idx2op: Dict[int, NvOp] = {}
    for idx, op_name in enumerate(op_graph.nodes.keys()):
        op_name2idx[op_name] = idx
        idx2op[idx] = op_graph.nodes[op_name].op

    constraints = []
    for e in op_graph.edges:
        constraints.append((op_name2idx[e.src.node_id], op_name2idx[e.dst.node_id]))

    partitions = generate_all_partitions_with_constraints(
        len(op_graph.nodes),
        constraints,
        min_nstages,
        max_nstages,
        min_ops_per_stage,
        max_ops_per_stage,
    )

    plans = []

    for partition in partitions:
        stages = []
        for seg in partition:
            stage = NvOpSequence()
            for idx in seg:
                stage.append(idx2op[idx])
            stages.append(stage)

        for shifts in product(range(min_shift, max_shift + 1), repeat=len(stages) - 1):
            if validator.validate(PipelinePlan(stages, shifts)):
                plan = PipelinePlan(stages, shifts)
                plans.append(plan)

    end_time = time.time()
    elapsed_ms = (end_time - start_time) * 1000
    print(f"enumerate_pipeline_plans took {elapsed_ms:.2f} ms")
    return plans
