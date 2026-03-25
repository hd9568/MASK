from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Union, Sequence, Tuple, Optional

from sparsene.op_gen.nvir.nvop import NvOpProgram, ForLoopNvOp, NvOpSequence, NvOp
from sparsene.op_gen.nvir.plan import apply_pipeline, PipelinePlan
from sparsene.op_gen.nvir.codegen import NvIrCodeGenerator
from sparsene.logging import get_logger

logger = get_logger(__name__)

NAIVE_FUSION_KINDS = {
    "add",
    "sub",
    "mul",
    "div",
    "pow",
    "const",
}


def _get_logical_kind(op: NvOp) -> str:
    explicit_kind = getattr(op, "logical_kind", None)
    if isinstance(explicit_kind, str) and explicit_kind:
        return explicit_kind

    attrs = getattr(op, "attrs", {})
    if isinstance(attrs, dict):
        attr_kind = attrs.get("logical_kind")
        if isinstance(attr_kind, str) and attr_kind:
            return attr_kind

    op_name = getattr(op, "name", "")
    if isinstance(op_name, str) and "const" in op_name.lower():
        return "const"

    return "unknown"


def _preferred_stage(kind: str) -> int:
    if kind in {
        "coo_atomic_format_load_idx",
        "coo_atomic_format_load_val",
        "mco_atomic_format_load_mask",
        "mco_atomic_format_load_val",
        "array_ref",
        "add",
        "sub",
        "mul",
        "div",
        "pow",
        "const",
        "load_offset",
        "load",
    }:
        return 0
    if kind in {"coo_atomic_val_restore", "mco_atomic_val_restore"}:
        return 1
    if kind in {"ldmatrix", "mma", "c_val_store"}:
        return 2
    return 0


def _collect_local_deps(main_loop: ForLoopNvOp) -> Dict[NvOp, Set[NvOp]]:
    local_ops = set(main_loop.body.ops)
    deps: Dict[NvOp, Set[NvOp]] = {}
    for op in main_loop.body.ops:
        producers: Set[NvOp] = set()
        for inp in op.inputs:
            source = inp.tensor.source
            if hasattr(source, "op") and source is not None:
                producer = getattr(source, "op", None)
                if producer in local_ops and producer is not op:
                    producers.add(producer)
        deps[op] = producers
    return deps


def _stable_topological_ops(main_loop: ForLoopNvOp, deps: Dict[NvOp, Set[NvOp]]) -> List[NvOp]:
    ops = list(main_loop.body.ops)
    original_index = {op: idx for idx, op in enumerate(ops)}
    indegree = {op: len(deps[op]) for op in ops}
    adjacency = defaultdict(list)
    for consumer, producers in deps.items():
        for producer in producers:
            adjacency[producer].append(consumer)

    ready = [op for op in ops if indegree[op] == 0]
    ready.sort(key=lambda op: original_index[op])

    ordered = []
    while ready:
        current = ready.pop(0)
        ordered.append(current)
        for nxt in adjacency[current]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                ready.append(nxt)
        ready.sort(key=lambda op: original_index[op])

    if len(ordered) != len(ops):
        logger.warning(
            "Dependency graph contains cycles or unresolved edges; fallback to original op order for pipeline partitioning."
        )
        return ops
    return ordered


def _collect_local_consumers(deps: Dict[NvOp, Set[NvOp]]) -> Dict[NvOp, Set[NvOp]]:
    consumers: Dict[NvOp, Set[NvOp]] = {op: set() for op in deps.keys()}
    for consumer, producers in deps.items():
        for producer in producers:
            if producer in consumers:
                consumers[producer].add(consumer)
    return consumers


def _build_fused_groups(
    ordered_ops: List[NvOp],
    deps: Dict[NvOp, Set[NvOp]],
) -> Tuple[
    List[List[NvOp]],
    Dict[int, Set[int]],
    Dict[int, int],
    Dict[NvOp, int],
    List[Tuple[str, str]],
]:
    original_index = {op: idx for idx, op in enumerate(ordered_ops)}
    consumers = _collect_local_consumers(deps)
    fusion_pairs: List[Tuple[str, str]] = []

    parent: Dict[NvOp, NvOp] = {op: op for op in ordered_ops}

    def find(op: NvOp) -> NvOp:
        while parent[op] is not op:
            parent[op] = parent[parent[op]]
            op = parent[op]
        return op

    def union(a: NvOp, b: NvOp) -> None:
        root_a = find(a)
        root_b = find(b)
        if root_a is root_b:
            return
        parent[root_a] = root_b
        fusion_pairs.append((a.name, b.name))

    for op in ordered_ops:
        kind = _get_logical_kind(op)
        if kind not in NAIVE_FUSION_KINDS:
            continue

        op_consumers = sorted(
            [consumer for consumer in consumers.get(op, set()) if consumer in original_index],
            key=lambda consumer: original_index[consumer],
        )
        if len(op_consumers) != 1:
            continue

        union(op, op_consumers[0])

    root_to_gid: Dict[NvOp, int] = {}
    groups: List[List[NvOp]] = []
    op_to_gid: Dict[NvOp, int] = {}

    for op in ordered_ops:
        root = find(op)
        if root not in root_to_gid:
            root_to_gid[root] = len(groups)
            groups.append([])
        gid = root_to_gid[root]
        groups[gid].append(op)
        op_to_gid[op] = gid

    group_deps: Dict[int, Set[int]] = {gid: set() for gid in range(len(groups))}
    for consumer, producers in deps.items():
        consumer_gid = op_to_gid[consumer]
        for producer in producers:
            producer_gid = op_to_gid[producer]
            if producer_gid != consumer_gid:
                group_deps[consumer_gid].add(producer_gid)

    group_preferred: Dict[int, int] = {}
    for gid, group_ops in enumerate(groups):
        kinds = [_get_logical_kind(op) for op in group_ops]
        non_naive_prefs = [
            _preferred_stage(kind) for kind in kinds if kind not in NAIVE_FUSION_KINDS
        ]
        if non_naive_prefs:
            group_preferred[gid] = max(non_naive_prefs)
        else:
            group_preferred[gid] = max((_preferred_stage(kind) for kind in kinds), default=0)

    return groups, group_deps, group_preferred, op_to_gid, fusion_pairs


def _stable_topological_groups(group_deps: Dict[int, Set[int]]) -> List[int]:
    ngroups = len(group_deps)
    indegree = {gid: len(preds) for gid, preds in group_deps.items()}
    adjacency: Dict[int, List[int]] = defaultdict(list)

    for gid, preds in group_deps.items():
        for pred_gid in preds:
            adjacency[pred_gid].append(gid)

    ready = [gid for gid in range(ngroups) if indegree.get(gid, 0) == 0]
    ready.sort()

    ordered: List[int] = []
    while ready:
        current = ready.pop(0)
        ordered.append(current)
        for nxt in adjacency.get(current, []):
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                ready.append(nxt)
        ready.sort()

    if len(ordered) != ngroups:
        logger.warning("Group graph is not a DAG; fallback to creation order.")
        return list(range(ngroups))

    return ordered


def _compress_stage_ids(assignment: Dict[int, int]) -> Dict[int, int]:
    used_stages = sorted(set(assignment.values()))
    remap = {old: new for new, old in enumerate(used_stages)}
    return {gid: remap[stage] for gid, stage in assignment.items()}


def _assign_group_stages(
    group_order: List[int],
    group_deps: Dict[int, Set[int]],
    group_preferred: Dict[int, int],
) -> Dict[int, int]:
    if not group_order:
        return {}

    max_stage_limit = max(2, min(8, len(group_order) + 1))

    best_assignment: Optional[Dict[int, int]] = None
    best_score: Optional[int] = None

    for stage_limit in range(1, max_stage_limit + 1):
        assignment: Dict[int, int] = {}

        def candidate_stages(gid: int) -> List[int]:
            preds = group_deps.get(gid, set())
            if not preds:
                values = list(range(stage_limit))
            else:
                lower = max(assignment[pred] for pred in preds)
                upper = min(assignment[pred] + 1 for pred in preds)
                lower = max(0, lower)
                upper = min(stage_limit - 1, upper)
                if lower > upper:
                    return []
                values = list(range(lower, upper + 1))

            preferred = group_preferred.get(gid, 0)
            values.sort(key=lambda stage: (abs(stage - preferred), stage))
            return values

        def backtrack(idx: int) -> bool:
            if idx >= len(group_order):
                return True
            gid = group_order[idx]
            for stage in candidate_stages(gid):
                assignment[gid] = stage
                if backtrack(idx + 1):
                    return True
            assignment.pop(gid, None)
            return False

        if backtrack(0):
            compact_assignment = _compress_stage_ids(assignment)
            num_stages = max(compact_assignment.values()) + 1
            preference_mismatch = sum(
                abs(compact_assignment[gid] - group_preferred.get(gid, 0))
                for gid in group_order
            )
            score = preference_mismatch * 8 + (num_stages - 1)

            if best_score is None or score < best_score:
                best_score = score
                best_assignment = compact_assignment

    if best_assignment is not None:
        return best_assignment

    logger.warning(
        "Cannot satisfy strict adjacent-stage dependency constraints; falling back to monotonic stage assignment."
    )
    relaxed_assignment: Dict[int, int] = {}
    for gid in group_order:
        preds = group_deps.get(gid, set())
        earliest = max((relaxed_assignment[pred] for pred in preds), default=0)
        preferred = group_preferred.get(gid, 0)
        relaxed_assignment[gid] = max(earliest, preferred)
    return _compress_stage_ids(relaxed_assignment)


def _partition_pipeline_stages(main_loop: ForLoopNvOp) -> List[List[NvOp]]:
    deps = _collect_local_deps(main_loop)
    ordered_ops = _stable_topological_ops(main_loop, deps)
    groups, group_deps, group_preferred, op_to_gid, fusion_pairs = _build_fused_groups(ordered_ops, deps)
    group_order = _stable_topological_groups(group_deps)
    group_stage = _assign_group_stages(group_order, group_deps, group_preferred)

    if not group_stage:
        return [ordered_ops] if ordered_ops else []

    num_stages = max(group_stage.values()) + 1
    stage_ops: List[List[NvOp]] = [[] for _ in range(num_stages)]
    for op in ordered_ops:
        gid = op_to_gid[op]
        stage_ops[group_stage[gid]].append(op)

    stage_sizes = ", ".join(
        f"S{idx}({len(ops)})" for idx, ops in enumerate(stage_ops)
    )
    logger.info(
        "Dynamic partition summary: total_ops=%d, fused_groups=%d, stages=%d [%s]",
        len(ordered_ops),
        len(groups),
        len(stage_ops),
        stage_sizes,
    )

    if fusion_pairs:
        fusion_desc = ", ".join(f"{src}->{dst}" for src, dst in fusion_pairs)
        logger.info("Naive-op fusion edges: %s", fusion_desc)
    else:
        logger.info("Naive-op fusion edges: none")

    for gid, group_ops in enumerate(groups):
        stage_id = group_stage.get(gid, 0)
        preferred = group_preferred.get(gid, 0)
        op_desc = ", ".join(
            f"{op.name}({_get_logical_kind(op)})" for op in group_ops
        )
        dep_desc = ", ".join(str(dep_gid) for dep_gid in sorted(group_deps.get(gid, set())))
        if not dep_desc:
            dep_desc = "none"
        logger.info(
            "Group G%d -> Stage%d (preferred=%d, deps=[%s]): %s",
            gid,
            stage_id,
            preferred,
            dep_desc,
            op_desc,
        )

    for stage_id, ops in enumerate(stage_ops):
        stage_op_names = ", ".join(op.name for op in ops)
        logger.info("Stage%d ops: %s", stage_id, stage_op_names)

    return stage_ops


def _collect_for_loops_recursive(
    ops: Sequence[NvOp],
    depth: int = 0,
) -> List[Tuple[ForLoopNvOp, int]]:
    collected: List[Tuple[ForLoopNvOp, int]] = []
    for op in ops:
        if not isinstance(op, ForLoopNvOp):
            continue
        collected.append((op, depth))
        collected.extend(_collect_for_loops_recursive(op.body.ops, depth + 1))
    return collected


def _select_main_loop(program: NvOpProgram) -> Optional[ForLoopNvOp]:
    loops_with_depth = _collect_for_loops_recursive(program.ops)
    if not loops_with_depth:
        return None

    candidate_loops: List[Tuple[ForLoopNvOp, int]] = []
    for loop, depth in loops_with_depth:
        if loop.blk_idx_mapping is not None:
            continue
        if len(loop.body.ops) == 0:
            continue
        candidate_loops.append((loop, depth))

    if not candidate_loops:
        return loops_with_depth[0][0]

    def _score(item: Tuple[ForLoopNvOp, int]) -> Tuple[int, int, int]:
        loop, depth = item
        non_loop_ops = sum(1 for body_op in loop.body.ops if not isinstance(body_op, ForLoopNvOp))
        return (depth, non_loop_ops, len(loop.body.ops))

    return max(candidate_loops, key=_score)[0]

def apply_software_pipeline_and_codegen(program: NvOpProgram, output_path: Union[str, Path]):
    """
    自动寻找主循环，划分流水线阶段，并生成 C++ 代码。
    """
    # 1. 寻找主循环 (Main Loop)
    # 递归查找并优先最内层、非 blockIdx 映射的计算循环，避免误选外层 blk_x/blk_y 包装循环。
    main_loop = _select_main_loop(program)
            
    if main_loop is None:
        logger.warning("No ForLoopNvOp found in the program. Skipping pipeline application.")
        _dump_code(program, output_path)
        return

    logger.info(
        "Selected main loop for pipeline: %s (blk_idx_mapping=%s, body_ops=%d)",
        main_loop.name,
        main_loop.blk_idx_mapping,
        len(main_loop.body.ops),
    )

    # 2. 按依赖关系进行动态 Stage 划分（满足 producer->consumer 仅允许同 stage 或下一 stage）
    stage_op_lists = _partition_pipeline_stages(main_loop)
    stage_sequences = [NvOpSequence(*stage_ops) for stage_ops in stage_op_lists if stage_ops]

    if not stage_sequences:
        logger.warning("No ops found in selected main loop body. Skipping pipeline application.")
        _dump_code(program, output_path)
        return

    stage_summary = ", ".join(
        f"Stage{idx}({len(stage.ops)} ops)" for idx, stage in enumerate(stage_sequences)
    )
    logger.info("Pipeline stages partitioned dynamically: %s", stage_summary)

    # 3. 仅当 stage>=2 才应用软件流水线；单 stage 场景直接按顺序代码生成
    if len(stage_sequences) >= 2:
        shifts = [1] * (len(stage_sequences) - 1)
        plan = PipelinePlan(stage_sequences, shifts)
        apply_pipeline(main_loop, plan)
    else:
        logger.info("Single-stage schedule selected; skipping pipeline wrapping.")

    # 4. 生成并写入最终代码
    _dump_code(program, output_path)


def _dump_code(program: NvOpProgram, output_path: Union[str, Path]):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    codegen = NvIrCodeGenerator()
    code_str = codegen.dump_nvop_program(program)
    
    with open(output_path, "w") as f:
        f.write(code_str)
        
    logger.info(f"Successfully dumped CUDA kernel to {output_path}")