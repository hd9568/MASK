"""Lower exported custom Format DSL designs to named CUDA operator includes."""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from sparsene.formats.custom.loader import load_run_format
from sparsene.formats.dsl import build_format_from_spec, validate_format_spec
from sparsene.op_gen.backend_registry import (
    materialize_backend_format_spec,
    select_backend,
)
from sparsene.op_gen.format_portfolio import select_portfolio_candidate
from sparsene.op_gen.computent.computent import computent_from_rts
from sparsene.op_gen.nvir.compiler_driver import apply_software_pipeline_and_codegen
from sparsene.op_gen.nvir.generate import generate_nvir
from sparsene.op_gen.opir.cValFlattenPass import CValFlattenPass
from sparsene.op_gen.opir.generate import generate_from_computent
from sparsene.op_gen.opir.varlenLoweringPass import VarlenLoweringPass
from sparsene.op_gen.strategy_agent import StrategyConfig, StrategyDecision
from sparsene.transform.rts import derive_rts


def _write_text(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        rendered = str(value)
    except Exception as exc:
        # Some legacy Computent printers depend on a process-global symbol
        # table that is intentionally reset after lowering.  Preserve a useful
        # structural representation instead of making audit logging fatal.
        rendered = f"<str unavailable: {type(exc).__name__}: {exc}>\n{value!r}"
    path.write_text(rendered + "\n", encoding="utf-8")


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return str(value)


def _specialize_single_warp_synchronization(operator_path: Path) -> int:
    """Lower block barriers for the generated operator's one-warp launch ABI.

    The current SM80 custom-format backend deliberately launches exactly one
    warp per CUDA block.  Keeping block-wide barriers in the emitted pipeline
    is correct but unnecessarily expensive, especially for short sparse-tile
    loops.  Make the launch requirement explicit in the include and use a warp
    barrier, which still provides the ordering required by the warp-issued
    cp.async/shared-memory pipeline.
    """

    source = operator_path.read_text(encoding="utf-8")
    marker_name = "#define SPARSENE_GENERATED_REQUIRES_SINGLE_WARP 1"
    if marker_name in source:
        if "__syncthreads();" in source:
            raise RuntimeError(
                f"{operator_path}: single-warp contract still contains block barriers"
            )
        # Exclude the macro definition itself and report lowered call sites so
        # existing audit manifests remain useful after lowering moved into
        # core NVIR codegen.
        return max(source.count("SPARSENE_GENERATED_BLOCK_SYNC()") - 1, 0)
    barrier_count = source.count("__syncthreads();")
    if barrier_count == 0:
        raise RuntimeError(f"{operator_path}: generated operator has no block barriers")
    marker = (
        marker_name + "\n"
        "#define SPARSENE_GENERATED_THREADS_PER_BLOCK 32\n"
        "#define SPARSENE_GENERATED_BLOCK_SYNC() __syncwarp()\n"
    )
    anchor = "#define VARLEN 0\n"
    if source.count(anchor) != 1:
        raise RuntimeError(f"{operator_path}: cannot locate generated ABI marker")
    source = source.replace(anchor, anchor + "\n" + marker, 1)
    source = source.replace("__syncthreads();", "SPARSENE_GENERATED_BLOCK_SYNC();")
    operator_path.write_text(source, encoding="utf-8")
    return barrier_count


def lower_format_operator(
    *,
    format_obj,
    codegen_format_name: str,
    operator_path: Path,
    ir_dir: Path,
    strategy_dir: Path,
    strategy_decision: StrategyDecision | None = None,
) -> dict[str, Any]:
    """Lower any executable Format object through the shared generated path.

    Keeping this independent of the format-design agent is important for the
    built-in backend portfolio: SR-BCRS, BIT-TCF, and future executable
    backends must exercise exactly the same Format -> CUDA middle end as a
    custom proposal.
    """
    ir_dir.mkdir(parents=True, exist_ok=True)
    operator_path.parent.mkdir(parents=True, exist_ok=True)

    rts = derive_rts(format_obj)
    _write_text(ir_dir / "01_rts.txt", rts)
    computent = computent_from_rts(codegen_format_name, rts)
    _write_text(ir_dir / "02_computent.txt", computent)
    opir = generate_from_computent(computent)
    _write_text(ir_dir / "03_opir.txt", opir)
    lowered = VarlenLoweringPass(
        op_builder=None,
        varlen2LenArrayTable=computent.varlen2LenArrayTable,
    ).run(opir)
    _write_text(ir_dir / "04_varlen_lowered_opir.txt", lowered)
    flattened = CValFlattenPass(op_builder=None).run(lowered)
    _write_text(ir_dir / "05_cval_flattened_opir.txt", flattened)
    program = generate_nvir(
        opir=flattened,
        format_name=codegen_format_name,
        strategy_config=(
            StrategyConfig(provider="heuristic", log_dir=str(strategy_dir))
            if strategy_decision is None
            else None
        ),
        strategy_decision_override=strategy_decision,
        strategy_log_dir_override=(
            str(strategy_dir) if strategy_decision is not None else None
        ),
    )
    _write_text(ir_dir / "06_nvir_before_pipeline.txt", program)
    apply_software_pipeline_and_codegen(program, operator_path)
    lowered_barriers = _specialize_single_warp_synchronization(operator_path)
    return {
        "operator_sha256": hashlib.sha256(operator_path.read_bytes()).hexdigest(),
        "lowered_block_barriers": lowered_barriers,
        "strategy_decision": _jsonable(program.attrs.get("strategy_decision")),
        "gmem_inouts": {
            name: {"dtype": str(value.dtype), "shape": str(value.shape)}
            for name, value in program.gmem_inouts.items()
        },
    }


def generate_one(
    run_dir: Path,
    dataset: str,
    *,
    tile_n: int = 64,
    max_tiles_per_execution_segment: int = 32,
    candidate_id: str = "agent",
    worktree_label: str | None = None,
    strategy_decision: StrategyDecision | None = None,
    work_dir_override: Path | None = None,
) -> dict[str, Any]:
    spec, _format_obj, format_artifact_path, format_artifact_sha256 = load_run_format(
        run_dir, dataset
    )
    candidate = select_portfolio_candidate(spec, candidate_id)
    plan = select_backend(
        candidate.spec,
        target="sm80",
        tile_n=tile_n,
        max_tiles_per_execution_segment=max_tiles_per_execution_segment,
    )
    physical_spec = materialize_backend_format_spec(candidate.spec, plan)
    physical_validation = validate_format_spec(physical_spec)
    if physical_validation["status"] != "valid":
        raise RuntimeError(
            "backend physicalization produced invalid Format DSL: "
            + "; ".join(physical_validation["errors"])
        )
    physical_format_obj = build_format_from_spec(physical_spec)
    dataset_dir = run_dir / "datasets" / dataset
    if work_dir_override is not None:
        work_dir = work_dir_override.resolve()
        operator_dir = work_dir / "operators"
        label = worktree_label or work_dir.name
        operator_name = f"{dataset}_{run_dir.name}_{label}.inc"
    elif candidate_id == "agent" and worktree_label is None:
        work_dir = dataset_dir / "operator_generation"
        operator_dir = dataset_dir / "operators"
        operator_name = f"{dataset}_{run_dir.name}.inc"
    else:
        label = worktree_label or candidate_id
        work_dir = dataset_dir / "operator_generation" / "portfolio" / label
        operator_dir = work_dir / "operators"
        operator_name = f"{dataset}_{run_dir.name}_{label}.inc"
    ir_dir = work_dir / "ir"
    operator_path = operator_dir / operator_name
    strategy_dir = work_dir / "strategy_agent"
    ir_dir.mkdir(parents=True, exist_ok=True)
    operator_dir.mkdir(parents=True, exist_ok=True)

    lowering = lower_format_operator(
        format_obj=physical_format_obj,
        codegen_format_name=plan.codegen_format_name,
        operator_path=operator_path,
        ir_dir=ir_dir,
        strategy_dir=strategy_dir,
        strategy_decision=strategy_decision,
    )
    manifest = {
        "schema_version": "sparsene.operator_generation.v1",
        "dataset": dataset,
        "format_name": candidate.spec["format_name"],
        "format_source": {
            "run_id": run_dir.name,
            "artifact_path": str(format_artifact_path),
            "artifact_sha256": format_artifact_sha256,
            "schema_version": spec.get("schema_version"),
            "request_id": spec.get("request_id"),
        },
        "portfolio_candidate": candidate.provenance(),
        "backend_plan": plan.to_dict(),
        "backend_physicalization": {
            "applied": physical_spec != candidate.spec,
            "packing_mode": plan.packing_mode,
            "physical_format_name": physical_spec["format_name"],
            "physical_spec_sha256": hashlib.sha256(
                json.dumps(
                    physical_spec, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                ).encode("utf-8")
            ).hexdigest(),
            "physical_spec": physical_spec if physical_spec != candidate.spec else None,
        },
        "strategy_decision": lowering["strategy_decision"],
        "gmem_inouts": lowering["gmem_inouts"],
        "strategy_source": (
            "explicit_override" if strategy_decision is not None else "heuristic"
        ),
        "operator_path": str(operator_path.resolve()),
        "operator_sha256": lowering["operator_sha256"],
        "launch_contract": {
            "threads_per_block": 32,
            "warps_per_block": 1,
            "synchronization": "warp",
            "lowered_block_barriers": lowering["lowered_block_barriers"],
        },
        "performance_contract": "sparsene.kernel_only_post_conversion.v1",
        "work_dir": str(work_dir.resolve()),
        "status": "generated",
    }
    manifest_path = work_dir / "generation_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--first", type=int)
    parser.add_argument("--tile-n", type=int, default=64)
    parser.add_argument("--max-tiles-per-segment", type=int, default=32)
    parser.add_argument(
        "--candidate-id",
        default="agent",
        help="auditable executable portfolio candidate (default: original Agent proposal)",
    )
    parser.add_argument(
        "--worktree-label",
        help="isolate output below operator_generation/portfolio without changing candidate provenance",
    )
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    run_manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    datasets = args.datasets or list(run_manifest["datasets"])
    if args.first is not None:
        datasets = datasets[: args.first]
    for dataset in datasets:
        manifest = generate_one(
            run_dir,
            dataset,
            tile_n=args.tile_n,
            max_tiles_per_execution_segment=args.max_tiles_per_segment,
            candidate_id=args.candidate_id,
            worktree_label=args.worktree_label,
        )
        print(f"[generated] {dataset}: {manifest['operator_path']}")


if __name__ == "__main__":
    main()
