"""Generate, compile, run, verify, and benchmark exported custom formats.

This command is entirely offline: the saved format proposals and the heuristic
strategy backend are used; no LLM/API call is made.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import csv
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from generate_custom_format_operators import generate_one


REPO_ROOT = Path(__file__).resolve().parents[2]
TESTBED = REPO_ROOT / "examples/src_fp32/acc/testbed"


def _work_dir(run_dir: Path, manifest: dict[str, Any]) -> Path:
    """Resolve the manifest-owned work tree, including portfolio siblings."""
    value = manifest.get("work_dir")
    if value:
        return Path(value)
    return run_dir / "datasets" / manifest["dataset"] / "operator_generation"


def _run(
    command: list[str],
    log_path: Path,
    *,
    allowed_returncodes: tuple[int, ...] = (0,),
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(result.stdout, encoding="utf-8")
    if result.returncode not in allowed_returncodes:
        raise RuntimeError(
            f"command failed ({result.returncode}); see {log_path}: {' '.join(command)}"
        )
    return result


def compile_one(run_dir: Path, manifest: dict[str, Any], jobs: int) -> Path:
    dataset = manifest["dataset"]
    plan = manifest["backend_plan"]
    work_dir = _work_dir(run_dir, manifest)
    build_dir = work_dir / "build"
    reorder_codes = {None: 0, "row_rcm": 1, "first_col_sort": 2, "jaccard_greedy": 3}
    packing_codes = {
        "column_condensed": 0,
        "active_fixed_blocks": 1,
        "column_condensed_dense": 2,
        "active_fixed_blocks_dense": 3,
        "sparse_row_column_condensed_dense": 4,
    }
    try:
        reorder_code = reorder_codes[plan["row_reorder"]]
        packing_code = packing_codes[plan["packing_mode"]]
    except KeyError as exc:
        raise ValueError(f"host bridge does not implement backend plan field: {exc}") from exc
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
        f"-DSPARSENE_ROW_REORDER={reorder_code}",
        f"-DSPARSENE_PACKING_MODE={packing_code}",
        "-DSPARSENE_MAX_TILES_PER_SEGMENT="
        f"{plan.get('max_tiles_per_execution_segment', 32)}",
    ]
    compiler_strategy = manifest.get("strategy_decision", {}).get("compiler", {})
    max_registers = compiler_strategy.get("max_registers_per_thread")
    if max_registers is not None:
        configure.append(f"-DSPARSENE_MAX_REGISTERS_PER_THREAD={int(max_registers)}")
    (work_dir / "cmake_configure_command.json").write_text(
        json.dumps(configure, indent=2) + "\n", encoding="utf-8"
    )
    _run(configure, work_dir / "cmake_configure.log")
    build_command = [
        "cmake", "--build", str(build_dir), "--target", "custom-format-e2e", f"-j{jobs}"
    ]
    (work_dir / "compile_command.json").write_text(
        json.dumps(build_command, indent=2) + "\n", encoding="utf-8"
    )
    _run(build_command, work_dir / "compile.log")
    executable = build_dir / "custom-format-e2e"
    if not executable.is_file():
        raise RuntimeError(f"build succeeded but executable is missing: {executable}")
    return executable


def run_one(
    run_dir: Path,
    manifest: dict[str, Any],
    executable: Path,
    *,
    n: int,
    warmup: int,
    repeat: int,
    runtime_max_tiles_per_segment: int | None = None,
    execution_policy: str = "auto",
    task_group_tiles: int = 64,
    result_label: str | None = None,
) -> dict[str, Any]:
    dataset = manifest["dataset"]
    dataset_dir = run_dir / "datasets" / dataset
    work_dir = _work_dir(run_dir, manifest)
    features = json.loads((dataset_dir / "matrix_features.json").read_text(encoding="utf-8"))
    matrix_path = Path(features["path"])
    plan = manifest["backend_plan"]
    variant = (
        f"n{n}_tile{plan['tile_n']}_segment"
        f"{plan.get('max_tiles_per_execution_segment', 32)}"
    )
    if runtime_max_tiles_per_segment is not None:
        variant += (
            f"_runtime{runtime_max_tiles_per_segment}_{execution_policy}"
            f"_group{task_group_tiles}"
        )
    if result_label:
        safe_label = "".join(
            char if char.isalnum() or char in {"-", "_"} else "_"
            for char in result_label
        )
        variant += f"_{safe_label}"
    result_path = work_dir / "e2e_results" / f"{variant}.json"
    run_log = work_dir / "e2e_results" / f"{variant}.log"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(executable),
        "--filename",
        str(matrix_path),
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
        "--execution-policy",
        execution_policy,
        "--task-group-tiles",
        str(task_group_tiles),
    ]
    if runtime_max_tiles_per_segment is not None:
        command.extend(["--max-tiles-per-segment", str(runtime_max_tiles_per_segment)])
    (work_dir / "e2e_command.json").write_text(
        json.dumps(command, indent=2) + "\n", encoding="utf-8"
    )
    # The testbed deliberately returns 2 after it has written a complete JSON
    # record for a numerical mismatch.  Preserve that record so the feedback
    # controller can classify the candidate as incorrect instead of losing the
    # evidence behind a generic subprocess failure.
    result = _run(command, run_log, allowed_returncodes=(0, 2))
    if not result_path.is_file():
        raise RuntimeError(
            f"testbed returned {result.returncode} without result JSON; see {run_log}"
        )
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    payload["executable"] = str(executable.resolve())
    payload["operator_path"] = manifest["operator_path"]
    payload["backend_plan"] = plan
    payload["portfolio_candidate"] = manifest.get("portfolio_candidate")
    payload["runtime_execution"] = {
        "max_tiles_per_segment": runtime_max_tiles_per_segment,
        "execution_policy": execution_policy,
        "task_group_tiles": task_group_tiles,
    }
    payload["measurement_id"] = result_label
    payload["result_path"] = str(result_path.resolve())
    result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # Keep the historical single-result paths as pointers-by-copy to the most
    # recent variant while retaining every tuning observation above.
    (work_dir / "e2e_result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (work_dir / "run.log").write_text(result.stdout, encoding="utf-8")
    print(result.stdout.strip())
    return payload


def write_summary(run_dir: Path, results: list[dict[str, Any]]) -> None:
    merged: dict[str, dict[str, Any]] = {}
    for result_path in run_dir.glob("datasets/*/operator_generation/e2e_result.json"):
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
    results = sorted(merged.values(), key=lambda item: order.get(item["dataset"], len(order)))
    json_path = run_dir / "operator_e2e_summary.json"
    json_path.write_text(
        json.dumps(
            {
                "schema_version": "sparsene.custom_format_e2e_summary.v1",
                "performance_contract": "sparsene.kernel_only_post_conversion.v1",
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    with (run_dir / "operator_e2e_summary.csv").open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(
            out,
            fieldnames=[
                "dataset",
                "M",
                "K",
                "N",
                "nnz",
                "logical_blk_m",
                "physical_blk_m",
                "blk_k",
                "row_reorder",
                "packing_mode",
                "generated_ms",
                "cusparse_ms",
                "speedup_vs_cusparse",
                "correctness_passed",
                "mismatches",
                "relative_l2_error",
            ],
        )
        writer.writeheader()
        for item in results:
            writer.writerow(
                {
                    "dataset": item["dataset"],
                    **item["shape"],
                    **item["format"],
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
    parser.add_argument("--first", type=int)
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--tile-n", type=int, default=64)
    parser.add_argument("--max-tiles-per-segment", type=int, default=32)
    parser.add_argument(
        "--config-file",
        type=Path,
        help=(
            "optional JSON with default/datasets tile_n and "
            "max_tiles_per_execution_segment; datasets are still run serially"
        ),
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument(
        "--runtime-max-tiles-per-segment",
        type=int,
        help="override host execution segmentation; use 0 for automatic policy",
    )
    parser.add_argument(
        "--execution-policy",
        choices=("auto", "row", "strict_lb", "multi_bind"),
        default="auto",
    )
    parser.add_argument("--task-group-tiles", type=int, default=64)
    parser.add_argument("--skip-generate", action="store_true")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    run_manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    datasets = args.datasets or list(run_manifest["datasets"])
    if args.first is not None:
        datasets = datasets[: args.first]
    tuning_config: dict[str, Any] = {}
    if args.config_file is not None:
        tuning_config = json.loads(args.config_file.resolve().read_text(encoding="utf-8"))
        if tuning_config.get("schema_version") != "sparsene.custom_format_tuning.v1":
            raise ValueError(f"unsupported tuning config: {args.config_file}")
        if int(tuning_config.get("n", args.n)) != args.n:
            raise ValueError(
                f"config N={tuning_config.get('n')} does not match command N={args.n}"
            )
    default_tuning = tuning_config.get(
        "default",
        {
            "tile_n": args.tile_n,
            "max_tiles_per_execution_segment": args.max_tiles_per_segment,
        },
    )
    dataset_tuning = tuning_config.get("datasets", {})
    results: list[dict[str, Any]] = []
    for dataset in datasets:
        selected_tuning = dict(default_tuning)
        selected_tuning.update(dataset_tuning.get(dataset, {}))
        tile_n = int(selected_tuning["tile_n"])
        max_tiles_per_segment = int(
            selected_tuning["max_tiles_per_execution_segment"]
        )
        print(
            f"[serial-config] {dataset}: tile_n={tile_n} "
            f"max_tiles_per_segment={max_tiles_per_segment}",
            flush=True,
        )
        manifest_path = (
            run_dir / "datasets" / dataset / "operator_generation" / "generation_manifest.json"
        )
        if args.skip_generate:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            selected_plan = manifest["backend_plan"]
            if (
                int(selected_plan["tile_n"]) != tile_n
                or int(selected_plan.get("max_tiles_per_execution_segment", 32))
                != max_tiles_per_segment
            ):
                raise ValueError(
                    f"{dataset}: --skip-generate manifest does not match selected tuning"
                )
        else:
            generation_log = (
                run_dir
                / "datasets"
                / dataset
                / "operator_generation"
                / "generation_stdout.log"
            )
            generation_log.parent.mkdir(parents=True, exist_ok=True)
            with generation_log.open("w", encoding="utf-8") as log_out:
                with redirect_stdout(log_out), redirect_stderr(log_out):
                    manifest = generate_one(
                        run_dir,
                        dataset,
                        tile_n=tile_n,
                        max_tiles_per_execution_segment=max_tiles_per_segment,
                    )
        try:
            executable = compile_one(run_dir, manifest, args.jobs)
            payload = run_one(
                run_dir,
                manifest,
                executable,
                n=args.n,
                warmup=args.warmup,
                repeat=args.repeat,
                runtime_max_tiles_per_segment=args.runtime_max_tiles_per_segment,
                execution_policy=args.execution_policy,
                task_group_tiles=args.task_group_tiles,
            )
            manifest["status"] = (
                "validated" if payload["correctness"]["passed"] else "incorrect"
            )
            manifest["e2e_result"] = str(
                (run_dir / "datasets" / dataset / "operator_generation" / "e2e_result.json").resolve()
            )
            results.append(payload)
        except Exception as exc:
            manifest["status"] = "failed"
            manifest["error"] = str(exc)
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            write_summary(run_dir, results)
            raise
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        write_summary(run_dir, results)
    print(f"validated {len(results)} dataset(s); summary: {run_dir / 'operator_e2e_summary.json'}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"fatal: {exc}", file=sys.stderr)
        raise SystemExit(1)
