"""Generate and serially benchmark built-in executable backend candidates.

This deliberately bypasses both the format-design and strategy agents.  It is
used to validate the shared Format -> RTS -> Computent -> OPIR -> NVIR -> CUDA
middle end against known format families before format selection is tuned.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
from typing import Any

from generate_custom_format_operators import lower_format_operator
from sparsene.formats.SR_BCRS import SR_BCRS_FORMAT
from sparsene.op_gen.backend_registry import select_backend
from sparsene.utils.temp_paths import sparsekg_build_dir, sparsekg_toolchain_env


REPO_ROOT = Path(__file__).resolve().parents[2]
TESTBED = REPO_ROOT / "examples/src_fp32/acc/testbed"


def _sr_bcrs_spec() -> dict[str, Any]:
    return {
        "format_name": "SR_BCRS_16x8",
        "levels": [
            {
                "level": 1,
                "name": "row_window",
                "axes": [
                    {"name": "X_o", "type": "df", "direction": "ROW", "length": "M"},
                    {"name": "Y_s", "type": "sv", "direction": "COL", "length": "K"},
                    {"name": "X_i", "type": "df", "direction": "ROW", "length": "BLK_M"},
                ],
            },
            {
                "level": 2,
                "name": "condensed_block",
                "axes": [
                    {"name": "Y_so", "type": "df", "direction": "COL", "length": "K/BLK_K"},
                    {"name": "Y_si", "type": "df", "direction": "COL", "length": "BLK_K"},
                    {"name": "X_i", "type": "df", "direction": "ROW", "length": "BLK_M"},
                ],
            },
        ],
        "atomic": {
            "type": "dense",
            "x_axis": "X_i",
            "y_axis": "Y_si",
            "layout": "row_major",
        },
        "params": {"BLK_M": 16, "BLK_K": 8},
    }


def _run(command: list[str], log_path: Path) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=sparsekg_toolchain_env(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(result.stdout, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed ({result.returncode}); see {log_path}: {' '.join(command)}"
        )
    return result


def generate_candidate(base_dir: Path, *, tile_n: int, max_tiles: int) -> dict[str, Any]:
    plan = select_backend(
        _sr_bcrs_spec(),
        target="sm80",
        tile_n=tile_n,
        max_tiles_per_execution_segment=max_tiles,
    )
    operator_path = base_dir / "operators/sr_bcrs_16x8.inc"
    lowering = lower_format_operator(
        format_obj=SR_BCRS_FORMAT,
        codegen_format_name=plan.codegen_format_name,
        operator_path=operator_path,
        ir_dir=base_dir / "ir",
        strategy_dir=base_dir / "strategy_heuristic",
    )
    manifest = {
        "schema_version": "sparsene.builtin_backend_generation.v1",
        "candidate": "sr_bcrs_16x8",
        "format_source": "python/sparsene/formats/SR_BCRS.py::SR_BCRS_FORMAT",
        "agent_independent": True,
        "backend_plan": plan.to_dict(),
        "operator_path": str(operator_path.resolve()),
        "operator_sha256": lowering["operator_sha256"],
        "strategy_decision": lowering["strategy_decision"],
        "launch_contract": {
            "threads_per_block": 32,
            "warps_per_block": 1,
            "synchronization": "warp",
            "lowered_block_barriers": lowering["lowered_block_barriers"],
        },
        "performance_contract": "sparsene.kernel_only_post_conversion.v1",
    }
    (base_dir / "generation_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def compile_candidate(base_dir: Path, manifest: dict[str, Any]) -> Path:
    plan = manifest["backend_plan"]
    build_dir = sparsekg_build_dir(base_dir)
    configure = [
        "cmake",
        "-S",
        str(TESTBED),
        "-B",
        str(build_dir),
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DSPARSENE_GENERATED_OPERATOR={manifest['operator_path']}",
        f"-DSPARSENE_LOGICAL_BLK_M={plan['logical_blk_m']}",
        f"-DSPARSENE_PHYSICAL_BLK_M={plan['physical_blk_m']}",
        f"-DSPARSENE_BLK_K={plan['physical_blk_k']}",
        f"-DSPARSENE_TILE_N={plan['tile_n']}",
        "-DSPARSENE_ROW_REORDER=0",
        "-DSPARSENE_PACKING_MODE=2",
        f"-DSPARSENE_MAX_TILES_PER_SEGMENT={plan['max_tiles_per_execution_segment']}",
    ]
    _run(configure, base_dir / "cmake_configure.log")
    _run(
        ["cmake", "--build", str(build_dir), "--target", "custom-format-e2e", "-j1"],
        base_dir / "compile.log",
    )
    executable = build_dir / "custom-format-e2e"
    if not executable.is_file():
        raise RuntimeError(f"build succeeded but executable is missing: {executable}")
    return executable


def run_dataset(
    run_dir: Path,
    base_dir: Path,
    manifest: dict[str, Any],
    executable: Path,
    dataset: str,
    *,
    n: int,
    warmup: int,
    repeat: int,
    execution_policy: str,
    task_group_tiles: int,
) -> dict[str, Any]:
    dataset_dir = run_dir / "datasets" / dataset
    features = json.loads((dataset_dir / "matrix_features.json").read_text(encoding="utf-8"))
    result_dir = dataset_dir / "builtin_backend_e2e/sr_bcrs_16x8"
    result_dir.mkdir(parents=True, exist_ok=True)
    result_path = result_dir / f"n{n}_w{warmup}_r{repeat}.json"
    result = _run(
        [
            str(executable),
            "--filename",
            str(features["path"]),
            "--dataset",
            dataset,
            "--result-json",
            str(result_path),
            "--n",
            str(n),
            "--warmup",
            str(warmup),
            "--repeat",
            str(repeat),
            "--max-tiles-per-segment",
            "0",
            "--execution-policy",
            execution_policy,
            "--task-group-tiles",
            str(task_group_tiles),
        ],
        result_dir / f"n{n}_w{warmup}_r{repeat}.log",
    )
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    payload.update(
        {
            "candidate": manifest["candidate"],
            "backend_plan": manifest["backend_plan"],
            "operator_path": manifest["operator_path"],
            "operator_sha256": manifest["operator_sha256"],
            "executable": str(executable.resolve()),
        }
    )
    result_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (result_dir / "e2e_result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(result.stdout.strip(), flush=True)
    return payload


def write_summary(
    run_dir: Path,
    base_dir: Path,
    results: list[dict[str, Any]],
) -> None:
    merged: dict[str, dict[str, Any]] = {}
    for result_path in run_dir.glob(
        "datasets/*/builtin_backend_e2e/sr_bcrs_16x8/e2e_result.json"
    ):
        try:
            item = json.loads(result_path.read_text(encoding="utf-8"))
            merged[item["dataset"]] = item
        except (OSError, KeyError, json.JSONDecodeError):
            continue
    for item in results:
        merged[item["dataset"]] = item
    run_order = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))[
        "datasets"
    ]
    order = {dataset: index for index, dataset in enumerate(run_order)}
    results = sorted(
        merged.values(), key=lambda item: order.get(item["dataset"], len(order))
    )
    summary = {
        "schema_version": "sparsene.builtin_backend_e2e_summary.v1",
        "candidate": "sr_bcrs_16x8",
        "performance_contract": "sparsene.kernel_only_post_conversion.v1",
        "results": results,
    }
    (base_dir / "operator_e2e_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (base_dir / "operator_e2e_summary.csv").open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(
            out,
            fieldnames=[
                "dataset", "M", "K", "N", "nnz", "candidate", "packing_mode",
                "generated_ms", "cusparse_ms", "speedup_vs_cusparse",
                "correctness_passed", "mismatches", "relative_l2_error",
            ],
        )
        writer.writeheader()
        for item in results:
            writer.writerow(
                {
                    "dataset": item["dataset"],
                    **item["shape"],
                    "candidate": item["candidate"],
                    "packing_mode": item["format"]["packing_mode"],
                    "generated_ms": item["timing"]["generated_ms"],
                    "cusparse_ms": item["timing"]["cusparse_ms"],
                    "speedup_vs_cusparse": item["timing"]["speedup_vs_cusparse"],
                    "correctness_passed": item["correctness"]["passed"],
                    "mismatches": item["correctness"]["mismatches"],
                    "relative_l2_error": item["correctness"]["relative_l2_error"],
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--tile-n", type=int, default=64)
    parser.add_argument("--max-tiles-per-segment", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=32)
    parser.add_argument(
        "--execution-policy",
        choices=("auto", "row", "strict_lb", "multi_bind"),
        default="auto",
    )
    parser.add_argument("--task-group-tiles", type=int, default=64)
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--skip-compile", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    run_manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    datasets = args.datasets or list(run_manifest["datasets"])
    base_dir = run_dir / "builtin_backend_e2e/sr_bcrs_16x8"
    base_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = base_dir / "generation_manifest.json"
    if args.skip_generate:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = generate_candidate(
            base_dir,
            tile_n=args.tile_n,
            max_tiles=args.max_tiles_per_segment,
        )
    executable = base_dir / "build/custom-format-e2e"
    if not args.skip_compile:
        executable = compile_candidate(base_dir, manifest)
    if not executable.is_file():
        raise RuntimeError(f"missing executable: {executable}")
    if args.prepare_only:
        print(f"[prepared] {manifest['candidate']}: {executable}")
        return

    results = []
    for dataset in datasets:
        results.append(
            run_dataset(
                run_dir,
                base_dir,
                manifest,
                executable,
                dataset,
                n=args.n,
                warmup=args.warmup,
                repeat=args.repeat,
                execution_policy=args.execution_policy,
                task_group_tiles=args.task_group_tiles,
            )
        )
        write_summary(run_dir, base_dir, results)


if __name__ == "__main__":
    main()
