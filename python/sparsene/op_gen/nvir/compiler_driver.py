from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Union, Sequence, Tuple, Optional

from sparsene.op_gen.nvir.nvop import NvOpProgram, ForLoopNvOp, NvOpSequence, NvOp, NvOpImpl, NvOpInput, NvOpTensor, NvOpOutput
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


def _preferred_stage(kind: str, op: Optional[NvOp] = None) -> int:
    if kind == "array_ref" and op is not None:
        role = getattr(op, "attrs", {}).get("array_ref_role")
        if role == "b_val":
            return 1
        if role == "csr_row_ptr":
            # row_ptr copy is a no-op when restore uses flat scatter;
            # keep it in the same stage as csr_atomic_val_restore to
            # avoid an extra pipeline commit/wait barrier.
            return 1
    if kind in {
        "coo_atomic_format_load_idx",
        "coo_atomic_format_load_val",
        "mco_atomic_format_load_mask",
        "mco_atomic_format_load_val",
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
    if kind in {
        "coo_atomic_val_restore",
        "csr_atomic_val_restore",
        "ell_atomic_val_restore",
        "dia_atomic_val_restore",
        "mco_atomic_val_restore",
    }:
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
            _preferred_stage(group_ops[i].attrs.get("logical_kind", kinds[i]), group_ops[i])
            for i in range(len(kinds))
            if kinds[i] not in NAIVE_FUSION_KINDS
        ]
        if non_naive_prefs:
            group_preferred[gid] = max(non_naive_prefs)
        else:
            group_preferred[gid] = max(
                (_preferred_stage(group_ops[i].attrs.get("logical_kind", kinds[i]), group_ops[i]) for i in range(len(kinds))),
                default=0,
            )

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


def _fuse_load_offset_sub_div(program: NvOpProgram) -> None:
    """Inline gen_sub + gen_div into load_offset by changing out[1] to off_r-off_l.


    Before: load_offset(gmem, idx) → (off_l, off_r)
            off_r → gen_sub(off_r, off_l) → (r-l)
            (r-l) → gen_div((r-l), BLK_K) → 1*(r-l) → for_loop.r

    After:  load_offset(gmem, idx) → (off_l, off_r - off_l) → for_loop.r
            gen_sub and gen_div removed.

    This kills 2 template params from the outer for_loop and 2 ops from
    the non-pipelined computation path.
    """

    def _walk(body):
        to_remove = []
        for op in list(body.ops):
            if isinstance(op, ForLoopNvOp):
                _walk(op.body)
                continue

            kind = _get_logical_kind(op)
            if kind != "sub":
                continue
            gen_sub = op
            sub_inputs = list(gen_sub.inputs)
            if len(sub_inputs) != 2:
                continue
            sub_src0 = sub_inputs[0].tensor.source  # off_r
            sub_src1 = sub_inputs[1].tensor.source  # off_l
            if not isinstance(sub_src0, NvOpOutput) or not isinstance(sub_src1, NvOpOutput):
                continue
            if sub_src0.op is None or sub_src1.op is None:
                continue
            load_kind0 = _get_logical_kind(sub_src0.op)
            load_kind1 = _get_logical_kind(sub_src1.op)
            if load_kind0 != "load_offset" or load_kind1 != "load_offset":
                continue
            if sub_src0.op is not sub_src1.op:
                continue
            load_off = sub_src0.op

            gen_sub_out = gen_sub.outputs[0]
            gen_div = None
            for c in gen_sub_out.consumers:
                if _get_logical_kind(c) == "div":
                    gen_div = c
                    break
            if gen_div is None:
                continue

            logger.info("Fusing load_offset+gen_sub+gen_div chain")
            
            old_impl = load_off.impl.code_template
            # Rewrite second line: "r(0) = operand_0(operand_1 + 1);" → "r(0) = operand_0(operand_1 + 1) - l(0);"
            lines = old_impl.split("\n")
            lines[1] = lines[1].rstrip(";") + f" - {load_off.outputs[0].name}(0);"
            load_off.impl = NvOpImpl("\n".join(lines))

            div_out = gen_div.outputs[0]
            for c in list(div_out.consumers):
                for inp in c.inputs:
                    if inp.tensor.source is div_out:
                        inp.tensor.source = load_off.outputs[1]
                        inp.tensor.dtype = load_off.outputs[1].tensor.dtype
                        inp.tensor.shape = load_off.outputs[1].tensor.shape

            to_remove.extend([gen_sub, gen_div])

        for op in to_remove:
            if op in body.ops:
                body.ops.remove(op)

    for top_op in program.ops:
        if isinstance(top_op, ForLoopNvOp):
            _walk(top_op.body)
            continue
        name = getattr(top_op, "name", "")
        if hasattr(top_op, "body"):
            _walk(top_op.body)

    logger.info("Fused load_offset+gen_sub+gen_div: removed 2 ops from outer loop")


def _fuse_inline_gen_add(main_loop: ForLoopNvOp) -> bool:
    """Inline gen_add into array_ref and load_offset_1, eliminating 36 pipeline calls.

    Before:
        gen_add(off_l, l+{c}) → idx  (36 pipeline calls per nbuf iteration!)
        array_ref(val_sidx, idx) → sidx_smem
        load_offset_1(offset_arr, idx) → (ll, rr)

    After:
        array_ref(val_sidx, off_l, l+{c}) → internal: idx=off_l+lc
        load_offset_1(offset_arr, off_l, l+{c}) → internal: idx=off_l+lc
        gen_add REMOVED
    """

    body = main_loop.body

    gen_add_candidates = []
    arr_ref = None
    arr_ref2 = None
    load_off1 = None

    for op in list(body.ops):
        kind = _get_logical_kind(op)
        if kind == "add":
            gen_add_candidates.append(op)
        elif kind == "array_ref" and not op.name.startswith("array_ref_1") and not op.name.startswith("array_ref_2"):
            arr_ref = op
        elif kind == "array_ref" and op.name.startswith("array_ref_2"):
            arr_ref2 = op
        elif kind == "load_offset" and op.name.startswith("load_offset_"):
            load_off1 = op

    if not gen_add_candidates or arr_ref is None or load_off1 is None:
        return False

    gen_add = None
    ga_out = None
    for add_op in gen_add_candidates:
        candidate_out = add_op.outputs[0]
        arr_ref_uses_add = any(inp.tensor.source is candidate_out for inp in arr_ref.inputs)
        load_uses_add = any(inp.tensor.source is candidate_out for inp in load_off1.inputs)
        if not (arr_ref_uses_add and load_uses_add):
            continue
        expected_consumers = {arr_ref, load_off1}
        if arr_ref2 is not None and any(inp.tensor.source is candidate_out for inp in arr_ref2.inputs):
            expected_consumers.add(arr_ref2)
        if set(candidate_out.consumers) != expected_consumers:
            continue
        gen_add = add_op
        ga_out = candidate_out
        break

    if gen_add is None or ga_out is None:
        return False

    off_l_inp = gen_add.inputs[0]
    lc_inp = gen_add.inputs[1]

    # --- Fuse into array_ref ---
    new_ar: list = []
    for inp in arr_ref.inputs:
        if inp.tensor.source is ga_out:
            continue
        new_ar.append(NvOpInput(idx=len(new_ar), name=inp.name,
            tensor=NvOpTensor(shape=inp.tensor.shape, mem=inp.tensor.mem,
                              dtype=inp.tensor.dtype, source=inp.tensor.source)))
    new_ar.append(NvOpInput(idx=len(new_ar), name="off_l",
        tensor=NvOpTensor(shape=off_l_inp.tensor.shape, mem=off_l_inp.tensor.mem,
                          dtype=off_l_inp.tensor.dtype, source=off_l_inp.tensor.source)))
    new_ar.append(NvOpInput(idx=len(new_ar), name="lc",
        tensor=NvOpTensor(shape=lc_inp.tensor.shape, mem=lc_inp.tensor.mem,
                          dtype=lc_inp.tensor.dtype, source=lc_inp.tensor.source)))
    arr_ref.inputs = []
    for ni in new_ar:
        arr_ref.add_input(ni)
    arr_ref.impl = NvOpImpl(
        "int tile_len = size<0>(shape(result_0));\n"
        "        int idx = off_l + lc;\n"
        "        auto thr_tiler = make_shape(Int<4>{});\n"
        "        for (int i_load = lid; i_load * 4 < tile_len; i_load += 32) {\n"
        "            auto thr_coord = make_coord(i_load);\n"
        "            auto src = local_tile(operand_0(_, idx), thr_tiler, thr_coord);\n"
        "            auto dst = local_tile(result_0(_, buf_idx), thr_tiler, thr_coord);\n"
        "            sparsene_copy_g2s_128<int>(src, dst);\n"
        "        }")
    arr_ref.attrs["cp_async"] = True
    if arr_ref.outputs:
        arr_ref.outputs[0].attrs["cp_async"] = True
        arr_ref.outputs[0].attrs["align_16"] = True

    # --- Fuse into load_offset_1 ---
    if any(inp.tensor.source is ga_out for inp in load_off1.inputs):
        new_lo: list = []
        for inp in load_off1.inputs:
            if inp.tensor.source is ga_out:
                continue
            new_lo.append(NvOpInput(idx=len(new_lo), name=inp.name,
                tensor=NvOpTensor(shape=inp.tensor.shape, mem=inp.tensor.mem,
                                  dtype=inp.tensor.dtype, source=inp.tensor.source)))
        new_lo.append(NvOpInput(idx=len(new_lo), name="off_l",
            tensor=NvOpTensor(shape=off_l_inp.tensor.shape, mem=off_l_inp.tensor.mem,
                              dtype=off_l_inp.tensor.dtype, source=off_l_inp.tensor.source)))
        new_lo.append(NvOpInput(idx=len(new_lo), name="lc",
            tensor=NvOpTensor(shape=lc_inp.tensor.shape, mem=lc_inp.tensor.mem,
                              dtype=lc_inp.tensor.dtype, source=lc_inp.tensor.source)))
        load_off1.inputs = []
        for ni in new_lo:
            load_off1.add_input(ni)
        load_l_name = load_off1.outputs[0].name
        load_r_name = load_off1.outputs[1].name
        load_off1.impl = NvOpImpl(
            "int idx = off_l + lc;\n"
            f"        {load_l_name}(0, buf_idx) = operand_0(idx);\n"
            f"        {load_r_name}(0, buf_idx) = operand_0(idx + 1);")

    # --- Fuse into CSR row_ptr alias array_ref_2 ---
    if arr_ref2 is not None and any(inp.tensor.source is ga_out for inp in arr_ref2.inputs):
        new_ar2: list = []
        for inp in arr_ref2.inputs:
            if inp.tensor.source is ga_out:
                continue
            new_ar2.append(NvOpInput(idx=len(new_ar2), name=inp.name,
                tensor=NvOpTensor(shape=inp.tensor.shape, mem=inp.tensor.mem,
                                  dtype=inp.tensor.dtype, source=inp.tensor.source)))
        new_ar2.append(NvOpInput(idx=len(new_ar2), name="off_l",
            tensor=NvOpTensor(shape=off_l_inp.tensor.shape, mem=off_l_inp.tensor.mem,
                              dtype=off_l_inp.tensor.dtype, source=off_l_inp.tensor.source)))
        new_ar2.append(NvOpInput(idx=len(new_ar2), name="lc",
            tensor=NvOpTensor(shape=lc_inp.tensor.shape, mem=lc_inp.tensor.mem,
                              dtype=lc_inp.tensor.dtype, source=lc_inp.tensor.source)))
        arr_ref2.inputs = []
        for ni in new_ar2:
            arr_ref2.add_input(ni)
        arr_ref2.impl = NvOpImpl(
            "int idx = off_l + lc;\n"
            "        (void)idx;\n"
            "        //! array_ref lowered as logical view/alias")

    body.ops.remove(gen_add)
    logger.info("Inlined gen_add into array_ref/load_offset_1/(optional)array_ref_2; -1 pipeline op")
    return True


def _mark_codegen_inline_load_offset(main_loop: ForLoopNvOp) -> None:
    body = main_loop.body
    load_off1 = None
    coo_idx = None
    coo_val = None
    for op in body.ops:
        kind = _get_logical_kind(op)
        if kind == "load_offset" and op.name.startswith("load_offset_1"):
            load_off1 = op
        elif kind == "coo_atomic_format_load_idx":
            coo_idx = op
        elif kind == "coo_atomic_format_load_val":
            coo_val = op

    if load_off1 is None or coo_idx is None or coo_val is None:
        return
    if len(load_off1.outputs) < 2:
        return

    ll_out, rr_out = load_off1.outputs[0], load_off1.outputs[1]
    expected_consumers = {coo_idx, coo_val}
    if set(ll_out.consumers) != expected_consumers:
        return
    if set(rr_out.consumers) != expected_consumers:
        return

    load_off1.attrs["inline_scalar_outputs_codegen"] = True
    logger.info(
        "Marked %s for codegen-level scalar inlining into coo loaders",
        load_off1.name,
    )


def _reset_inputs(op: NvOp, new_inputs: List[NvOpInput]) -> None:
    for old_inp in op.inputs:
        source = old_inp.tensor.source
        if isinstance(source, NvOpOutput) and op in source.consumers:
            source.consumers.remove(op)
    op.inputs = []
    for new_inp in new_inputs:
        op.add_input(new_inp)


def _clone_input(inp: NvOpInput, *, idx: int, name: Optional[str] = None) -> NvOpInput:
    return NvOpInput(
        idx=idx,
        name=name or inp.name,
        tensor=NvOpTensor(
            shape=inp.tensor.shape,
            mem=inp.tensor.mem,
            dtype=inp.tensor.dtype,
            source=inp.tensor.source,
            row_major=inp.tensor.row_major,
            swizzle=inp.tensor.swizzle,
        ),
        layout_hint=inp.layout_hint,
    )


def _fuse_load_offset_into_coo_loaders(main_loop: ForLoopNvOp) -> None:
    body = main_loop.body

    load_off = None
    arr_ref = None
    coo_idx = None
    coo_val = None
    for op in list(body.ops):
        kind = _get_logical_kind(op)
        if kind == "load_offset" and op.name.startswith("load_offset_"):
            load_off = op
        elif kind == "array_ref" and not op.name.startswith("array_ref_1") and not op.name.startswith("array_ref_2"):
            arr_ref = op
        elif kind == "coo_atomic_format_load_idx":
            coo_idx = op
        elif kind == "coo_atomic_format_load_val":
            coo_val = op

    if load_off is None or arr_ref is None or coo_idx is None or coo_val is None:
        return
    if len(load_off.inputs) < 3 or len(load_off.outputs) < 2:
        return

    ll_out = load_off.outputs[0]
    rr_out = load_off.outputs[1]
    expected_consumers = {coo_idx, coo_val}
    if set(ll_out.consumers) != expected_consumers or set(rr_out.consumers) != expected_consumers:
        return

    offset_arr_inp = load_off.inputs[0]
    new_arr_inputs = [
        _clone_input(arr_ref.inputs[0], idx=0, name="operand_0"),
        _clone_input(offset_arr_inp, idx=1, name="offset_arr"),
        _clone_input(arr_ref.inputs[1], idx=2, name="off_l"),
        _clone_input(arr_ref.inputs[2], idx=3, name="lc"),
    ]
    _reset_inputs(arr_ref, new_arr_inputs)

    cloned_ll_out = NvOpOutput(
        idx=len(arr_ref.outputs),
        name="coo_ll",
        tensor=NvOpTensor(
            shape=ll_out.tensor.shape,
            mem=ll_out.tensor.mem,
            dtype=ll_out.tensor.dtype,
            row_major=ll_out.tensor.row_major,
            swizzle=ll_out.tensor.swizzle,
        ),
        origin=ll_out.origin,
        unique=ll_out.unique,
        layout_hint=ll_out.layout_hint,
        **ll_out.attrs,
    )
    cloned_rr_out = NvOpOutput(
        idx=len(arr_ref.outputs) + 1,
        name="coo_rr",
        tensor=NvOpTensor(
            shape=rr_out.tensor.shape,
            mem=rr_out.tensor.mem,
            dtype=rr_out.tensor.dtype,
            row_major=rr_out.tensor.row_major,
            swizzle=rr_out.tensor.swizzle,
        ),
        origin=rr_out.origin,
        unique=rr_out.unique,
        layout_hint=rr_out.layout_hint,
        **rr_out.attrs,
    )
    arr_ref.add_output(cloned_ll_out)
    arr_ref.add_output(cloned_rr_out)

    arr_src_name = arr_ref.inputs[0].name
    arr_offset_name = arr_ref.inputs[1].name
    arr_off_l_name = arr_ref.inputs[2].name
    arr_lc_name = arr_ref.inputs[3].name
    arr_out_name = arr_ref.outputs[0].name
    arr_ll_name = arr_ref.outputs[1].name
    arr_rr_name = arr_ref.outputs[2].name
    arr_ref.impl = NvOpImpl(
        f"""int tile_len = size<0>(shape({arr_out_name}));
int idx = {arr_off_l_name} + {arr_lc_name};
{arr_ll_name}(0, buf_idx) = {arr_offset_name}(idx);
{arr_rr_name}(0, buf_idx) = {arr_offset_name}(idx + 1);
auto thr_tiler = make_shape(Int<4>{{}});
for (int i_load = lid; i_load * 4 < tile_len; i_load += 32) {{
    auto thr_coord = make_coord(i_load);
    auto src = local_tile({arr_src_name}(_, idx), thr_tiler, thr_coord);
    auto dst = local_tile({arr_out_name}(_, buf_idx), thr_tiler, thr_coord);
    sparsene_copy_g2s_128<int>(src, dst);
}}"""
    )

    def _rewrite_coo_scalar_inputs(op: NvOp) -> None:
        new_inputs: List[NvOpInput] = []
        for inp in op.inputs:
            if inp.tensor.source is ll_out:
                new_inputs.append(
                    NvOpInput(
                        idx=len(new_inputs),
                        name=inp.name,
                        tensor=NvOpTensor(
                            shape=cloned_ll_out.tensor.shape,
                            mem=cloned_ll_out.tensor.mem,
                            dtype=cloned_ll_out.tensor.dtype,
                            source=cloned_ll_out,
                            row_major=cloned_ll_out.tensor.row_major,
                            swizzle=cloned_ll_out.tensor.swizzle,
                        ),
                        layout_hint=inp.layout_hint,
                    )
                )
            elif inp.tensor.source is rr_out:
                new_inputs.append(
                    NvOpInput(
                        idx=len(new_inputs),
                        name=inp.name,
                        tensor=NvOpTensor(
                            shape=cloned_rr_out.tensor.shape,
                            mem=cloned_rr_out.tensor.mem,
                            dtype=cloned_rr_out.tensor.dtype,
                            source=cloned_rr_out,
                            row_major=cloned_rr_out.tensor.row_major,
                            swizzle=cloned_rr_out.tensor.swizzle,
                        ),
                        layout_hint=inp.layout_hint,
                    )
                )
            else:
                new_inputs.append(_clone_input(inp, idx=len(new_inputs)))
        _reset_inputs(op, new_inputs)

    _rewrite_coo_scalar_inputs(coo_idx)
    _rewrite_coo_scalar_inputs(coo_val)
    body.ops.remove(load_off)
    logger.info("Fused load_offset_1 into val_sidx array_ref; -1 pipeline op without adding coo stage dependency")


def apply_software_pipeline_and_codegen(program: NvOpProgram, output_path: Union[str, Path]):
    """
    自动寻找主循环，划分流水线阶段，并生成 C++ 代码。
    """
    # 1. 寻找主循环 (Main Loop)
    _fuse_load_offset_sub_div(program)
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

    # 2. 内联融合: gen_add → array_ref + load_offset_1
    # Only safe when gen_add feeds exactly one array_ref and one load_offset
    # that both consume the add output. CSR's no-op array_ref_2 does not fit.
    # Pair the two passes together: if fuse skips, mark must also skip
    # because it relies on the post-fuse op signature (3 inputs).
    fused = _fuse_inline_gen_add(main_loop)
    if fused:
        _mark_codegen_inline_load_offset(main_loop)

    # 3. 按依赖关系进行动态 Stage 划分（满足 producer->consumer 仅允许同 stage 或下一 stage）
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

    strategy = program.attrs.get("strategy_decision")
    pipeline_strategy = getattr(strategy, "pipeline", None)
    if pipeline_strategy is not None:
        main_loop.attrs["pipeline_strategy"] = pipeline_strategy

    # 3. 仅当 stage>=2 才应用软件流水线；单 stage 场景直接按顺序代码生成
    if len(stage_sequences) >= 2:
        shifts = [1] * (len(stage_sequences) - 1)
        if pipeline_strategy is not None and pipeline_strategy.stage_shifts:
            if len(pipeline_strategy.stage_shifts) == len(stage_sequences) - 1:
                shifts = list(pipeline_strategy.stage_shifts)
            else:
                logger.warning(
                    "Ignoring invalid strategy stage_shifts=%s for %d stages",
                    pipeline_strategy.stage_shifts,
                    len(stage_sequences),
                )
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
