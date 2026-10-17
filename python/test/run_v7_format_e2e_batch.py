#!/usr/bin/env python3
"""Serially generate, compile, validate, and time every saved run format.

Unlike the single-format runner, this driver records a row for every requested
format even when a later format is not executable by the current semantic
backend.  It never substitutes a different format for an unsupported DSL and
it deliberately allows exactly one GPU evaluation process at a time.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import traceback
from typing import Any

from generate_custom_format_operators import generate_one
from run_custom_format_e2e import compile_one, run_one
from sparsene.formats.custom.loader import load_run_format
from sparsene.op_gen.backend_registry import select_backend


STATUS_SCHEMA = "sparsene.v7_format_e2e_status.v1"


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_status(run_dir: Path, rows: list[dict[str, Any]], settings: dict[str, Any]) -> None:
    payload = {
        "schema_version": STATUS_SCHEMA,
        "performance_contract": "sparsene.kernel_only_post_conversion.v1",
        "settings": settings,
        "counts": {
            "total": len(rows),
            "validated": sum(row["status"] == "validated" for row in rows),
            "incorrect": sum(row["status"] == "incorrect" for row in rows),
            "unsupported": sum(row["status"] == "unsupported" for row in rows),
            "failed": sum(row["status"] == "failed" for row in rows),
        },
        "results": rows,
    }
    _write_json(run_dir / "operator_e2e_v7_all_status.json", payload)
    fields = [
        "dataset", "status", "failure_phase", "format_name", "atomic_type",
        "artifact_sha256", "backend", "codegen_format_name", "packing_mode",
        "logical_blk_m", "logical_blk_k", "generated_ms", "cusparse_ms",
        "speedup_vs_cusparse", "correctness_passed", "mismatches",
        "relative_l2_error", "generation_manifest", "e2e_result", "error",
    ]
    with (run_dir / "operator_e2e_v7_all_status.csv").open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _record_base(run_dir: Path, dataset: str) -> tuple[dict[str, Any], dict[str, Any]]:
    spec, _fmt, artifact_path, artifact_sha = load_run_format(run_dir, dataset)
    return spec, {
        "dataset": dataset,
        "status": "failed",
        "failure_phase": "",
        "format_name": spec.get("format_name"),
        "atomic_type": spec.get("atomic", {}).get("type"),
        "artifact_path": str(artifact_path),
        "artifact_sha256": artifact_sha,
        "backend": "",
        "codegen_format_name": "",
        "packing_mode": "",
        "logical_blk_m": spec.get("params", {}).get("BLK_M"),
        "logical_blk_k": spec.get("params", {}).get("BLK_K"),
        "generated_ms": "",
        "cusparse_ms": "",
        "speedup_vs_cusparse": "",
        "correctness_passed": "",
        "mismatches": "",
        "relative_l2_error": "",
        "generation_manifest": "",
        "e2e_result": "",
        "error": "",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--tile-n", type=int, default=64)
    parser.add_argument("--max-tiles-per-segment", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=32)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--runtime-max-tiles-per-segment", type=int, default=0)
    parser.add_argument("--execution-policy", choices=("auto", "row", "strict_lb", "multi_bind"), default="auto")
    parser.add_argument("--task-group-tiles", type=int, default=64)
    args = parser.parse_args()
    if args.jobs != 1:
        raise SystemExit("This batch intentionally requires --jobs 1: one GPU evaluation at a time")

    run_dir = args.run_dir.resolve()
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    datasets = args.datasets or list(manifest["datasets"])
    settings = {
        "n": args.n, "tile_n": args.tile_n,
        "max_tiles_per_execution_segment": args.max_tiles_per_segment,
        "warmup": args.warmup, "repeat": args.repeat, "jobs": args.jobs,
        "runtime_max_tiles_per_segment": args.runtime_max_tiles_per_segment,
        "execution_policy": args.execution_policy,
        "task_group_tiles": args.task_group_tiles,
    }
    rows: list[dict[str, Any]] = []
    for index, dataset in enumerate(datasets, start=1):
        print(f"[{index:02d}/{len(datasets):02d}] {dataset}: serial validation", flush=True)
        try:
            spec, row = _record_base(run_dir, dataset)
        except Exception as exc:
            rows.append({"dataset": dataset, "status": "failed", "failure_phase": "artifact_load", "error": str(exc)})
            _write_status(run_dir, rows, settings)
            continue
        try:
            plan = select_backend(
                spec, target="sm80", tile_n=args.tile_n,
                max_tiles_per_execution_segment=args.max_tiles_per_segment,
            )
        except Exception as exc:
            row.update(status="unsupported", failure_phase="backend_selection", error=str(exc))
            rows.append(row)
            _write_status(run_dir, rows, settings)
            continue
        row.update(
            backend=plan.backend,
            codegen_format_name=plan.codegen_format_name,
            packing_mode=plan.packing_mode,
            logical_blk_m=plan.logical_blk_m,
            logical_blk_k=plan.logical_blk_k,
        )
        try:
            generated = generate_one(
                run_dir, dataset, tile_n=args.tile_n,
                max_tiles_per_execution_segment=args.max_tiles_per_segment,
            )
            row["generation_manifest"] = str(
                (run_dir / "datasets" / dataset / "operator_generation" / "generation_manifest.json").resolve()
            )
        except Exception as exc:
            row.update(status="failed", failure_phase="generation", error=str(exc))
            rows.append(row)
            _write_status(run_dir, rows, settings)
            continue
        try:
            executable = compile_one(run_dir, generated, args.jobs)
        except Exception as exc:
            row.update(status="failed", failure_phase="compile", error=str(exc))
            rows.append(row)
            _write_status(run_dir, rows, settings)
            continue
        try:
            result = run_one(
                run_dir, generated, executable, n=args.n, warmup=args.warmup,
                repeat=args.repeat,
                runtime_max_tiles_per_segment=args.runtime_max_tiles_per_segment,
                execution_policy=args.execution_policy,
                task_group_tiles=args.task_group_tiles,
            )
            correctness = result["correctness"]
            timing = result["timing"]
            row.update(
                status="validated" if correctness["passed"] else "incorrect",
                generated_ms=timing["generated_ms"],
                cusparse_ms=timing["cusparse_ms"],
                speedup_vs_cusparse=timing["speedup_vs_cusparse"],
                correctness_passed=correctness["passed"],
                mismatches=correctness["mismatches"],
                relative_l2_error=correctness["relative_l2_error"],
                e2e_result=str((Path(generated["work_dir"]) / "e2e_result.json").resolve()),
            )
        except Exception as exc:
            row.update(status="failed", failure_phase="run", error=str(exc))
        rows.append(row)
        _write_status(run_dir, rows, settings)
    _write_status(run_dir, rows, settings)
    counts = {status: sum(row.get("status") == status for row in rows) for status in ("validated", "incorrect", "unsupported", "failed")}
    print(f"complete: {counts}; {run_dir / 'operator_e2e_v7_all_status.json'}")
    return 0 if not counts["failed"] and not counts["incorrect"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
