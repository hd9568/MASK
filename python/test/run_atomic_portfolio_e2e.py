"""Serially measure an Agent format and semantic atomic siblings.

The input DSL is never overwritten.  Each candidate gets its own generated
include, build directory, logs, correctness result, and row in the portfolio
summary.  This is deliberately a one-GPU-process-at-a-time driver.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

from generate_custom_format_operators import generate_one
from run_custom_format_e2e import compile_one, run_one
from sparsene.formats.custom.loader import load_run_format
from sparsene.op_gen.format_portfolio import (
    executable_atomic_portfolio,
    executable_tcf_portfolio,
)


def _selected_tuning(
    config: dict[str, Any], dataset: str, tile_n: int, max_tiles: int
) -> tuple[int, int]:
    selected = dict(
        config.get(
            "default",
            {"tile_n": tile_n, "max_tiles_per_execution_segment": max_tiles},
        )
    )
    selected.update(config.get("datasets", {}).get(dataset, {}))
    return int(selected["tile_n"]), int(selected["max_tiles_per_execution_segment"])


def _write_summary(run_dir: Path, rows: list[dict[str, Any]]) -> None:
    rows.sort(key=lambda item: (item["dataset"], item["candidate_id"]))
    payload = {
        "schema_version": "sparsene.atomic_portfolio_e2e_summary.v1",
        "performance_contract": "sparsene.kernel_only_post_conversion.v1",
        "results": rows,
    }
    (run_dir / "atomic_portfolio_e2e_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    fields = [
        "dataset", "candidate_id", "candidate_source", "atomic_type",
        "codegen_format_name", "tile_n", "max_tiles_per_execution_segment",
        "generated_ms", "cusparse_ms", "speedup_vs_cusparse",
        "correctness_passed", "mismatches", "relative_l2_error", "operator_path",
    ]
    with (run_dir / "atomic_portfolio_e2e_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--first", type=int)
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--tile-n", type=int, default=64)
    parser.add_argument("--max-tiles-per-segment", type=int, default=32)
    parser.add_argument("--config-file", type=Path)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=32)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument(
        "--runtime-max-tiles-per-segment",
        type=int,
        default=0,
        help="host execution segmentation (default 0: automatic load-balanced policy)",
    )
    parser.add_argument(
        "--execution-policy",
        choices=("auto", "row", "strict_lb", "multi_bind"),
        default="auto",
    )
    parser.add_argument("--task-group-tiles", type=int, default=64)
    parser.add_argument(
        "--include-tile-siblings",
        action="store_true",
        help="also measure dense 16x8, 16x16, and 32x8 TCF siblings where valid",
    )
    args = parser.parse_args()
    if args.jobs != 1:
        raise ValueError("atomic portfolio evaluation is intentionally serial; use --jobs 1")

    run_dir = args.run_dir.resolve()
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    datasets = args.datasets or list(manifest["datasets"])
    if args.first is not None:
        datasets = datasets[: args.first]
    tuning: dict[str, Any] = {}
    if args.config_file is not None:
        tuning = json.loads(args.config_file.resolve().read_text(encoding="utf-8"))
        if tuning.get("schema_version") != "sparsene.custom_format_tuning.v1":
            raise ValueError(f"unsupported tuning config: {args.config_file}")
        if int(tuning.get("n", args.n)) != args.n:
            raise ValueError("tuning config N does not match --n")

    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        spec, _format_obj, _artifact_path, _artifact_sha256 = load_run_format(
            run_dir, dataset
        )
        tile_n, max_tiles = _selected_tuning(
            tuning, dataset, args.tile_n, args.max_tiles_per_segment
        )
        portfolio = (
            executable_tcf_portfolio(spec)
            if args.include_tile_siblings
            else executable_atomic_portfolio(spec)
        )
        for candidate in portfolio:
            # A unique worktree also protects the historical agent result from
            # being replaced by a different warmup/repeat configuration.
            worktree_label = f"atomic_portfolio_{candidate.candidate_id}"
            print(
                f"[serial-portfolio] {dataset} {candidate.candidate_id}: "
                f"tile_n={tile_n} max_tiles={max_tiles}",
                flush=True,
            )
            generated = generate_one(
                run_dir,
                dataset,
                tile_n=tile_n,
                max_tiles_per_execution_segment=max_tiles,
                candidate_id=candidate.candidate_id,
                worktree_label=worktree_label,
            )
            executable = compile_one(run_dir, generated, args.jobs)
            result = run_one(
                run_dir,
                generated,
                executable,
                n=args.n,
                warmup=args.warmup,
                repeat=args.repeat,
                runtime_max_tiles_per_segment=args.runtime_max_tiles_per_segment,
                execution_policy=args.execution_policy,
                task_group_tiles=args.task_group_tiles,
            )
            provenance = generated["portfolio_candidate"]
            rows.append(
                {
                    "dataset": dataset,
                    "candidate_id": provenance["candidate_id"],
                    "candidate_source": provenance["source"],
                    "atomic_type": provenance["atomic_type"],
                    "codegen_format_name": provenance["codegen_format_name"],
                    "tile_n": generated["backend_plan"]["tile_n"],
                    "max_tiles_per_execution_segment": generated["backend_plan"]["max_tiles_per_execution_segment"],
                    "generated_ms": result["timing"]["generated_ms"],
                    "cusparse_ms": result["timing"]["cusparse_ms"],
                    "speedup_vs_cusparse": result["timing"]["speedup_vs_cusparse"],
                    "correctness_passed": result["correctness"]["passed"],
                    "mismatches": result["correctness"]["mismatches"],
                    "relative_l2_error": result["correctness"]["relative_l2_error"],
                    "operator_path": generated["operator_path"],
                }
            )
            _write_summary(run_dir, rows)
    print(f"serially validated {len(rows)} portfolio candidate(s)")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"fatal: {exc}", file=sys.stderr)
        raise SystemExit(1)
