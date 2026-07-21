"""Lower exported custom Format DSL designs to named CUDA operator includes."""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from sparsene.formats.custom.loader import load_custom_format
from sparsene.op_gen.backend_registry import select_backend
from sparsene.op_gen.computent.computent import computent_from_rts
from sparsene.op_gen.nvir.compiler_driver import apply_software_pipeline_and_codegen
from sparsene.op_gen.nvir.generate import generate_nvir
from sparsene.op_gen.opir.cValFlattenPass import CValFlattenPass
from sparsene.op_gen.opir.generate import generate_from_computent
from sparsene.op_gen.opir.varlenLoweringPass import VarlenLoweringPass
from sparsene.op_gen.strategy_agent import StrategyConfig
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
        strategy_config=StrategyConfig(
            provider="heuristic",
            log_dir=str(strategy_dir),
        ),
    )
    _write_text(ir_dir / "06_nvir_before_pipeline.txt", program)
    apply_software_pipeline_and_codegen(program, operator_path)
    lowered_barriers = _specialize_single_warp_synchronization(operator_path)
    return {
        "operator_sha256": hashlib.sha256(operator_path.read_bytes()).hexdigest(),
        "lowered_block_barriers": lowered_barriers,
        "strategy_decision": _jsonable(program.attrs.get("strategy_decision")),
    }


def generate_one(
    run_dir: Path,
    dataset: str,
    *,
    tile_n: int = 64,
    max_tiles_per_execution_segment: int = 32,
) -> dict[str, Any]:
    spec, format_obj, _module = load_custom_format(dataset)
    if spec["provenance"]["run_id"] != run_dir.name:
        raise ValueError(
            f"{dataset}: exported module belongs to {spec['provenance']['run_id']}, "
            f"not {run_dir.name}"
        )
    plan = select_backend(
        spec,
        target="sm80",
        tile_n=tile_n,
        max_tiles_per_execution_segment=max_tiles_per_execution_segment,
    )
    dataset_dir = run_dir / "datasets" / dataset
    ir_dir = dataset_dir / "operator_generation" / "ir"
    operator_dir = dataset_dir / "operators"
    operator_path = operator_dir / f"{dataset}_{run_dir.name}.inc"
    strategy_dir = dataset_dir / "operator_generation" / "strategy_agent"
    ir_dir.mkdir(parents=True, exist_ok=True)
    operator_dir.mkdir(parents=True, exist_ok=True)

    lowering = lower_format_operator(
        format_obj=format_obj,
        codegen_format_name=plan.codegen_format_name,
        operator_path=operator_path,
        ir_dir=ir_dir,
        strategy_dir=strategy_dir,
    )
    manifest = {
        "schema_version": "sparsene.operator_generation.v1",
        "dataset": dataset,
        "format_name": spec["format_name"],
        "format_source": spec["provenance"],
        "backend_plan": plan.to_dict(),
        "strategy_decision": lowering["strategy_decision"],
        "operator_path": str(operator_path.resolve()),
        "operator_sha256": lowering["operator_sha256"],
        "launch_contract": {
            "threads_per_block": 32,
            "warps_per_block": 1,
            "synchronization": "warp",
            "lowered_block_barriers": lowering["lowered_block_barriers"],
        },
        "performance_contract": "sparsene.kernel_only_post_conversion.v1",
        "status": "generated",
    }
    manifest_path = dataset_dir / "operator_generation" / "generation_manifest.json"
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
        )
        print(f"[generated] {dataset}: {manifest['operator_path']}")


if __name__ == "__main__":
    main()
