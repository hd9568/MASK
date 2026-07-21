#!/usr/bin/env python3
"""Build the audited final table for the serial A800 N=512 run."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_DIR = (
    REPO_ROOT / "logs/e2e_runs/20260718_165146_deepseek_v4_pro_format26_v6"
)
DEFAULT_CONFIG = Path(__file__).with_name("a800_n512_fp32_final_config.json")
OUTPUT_CSV = Path(__file__).with_name("sparsene_a800_n512_fp32_final.csv")
OUTPUT_JSON = Path(__file__).with_name("sparsene_a800_n512_fp32_final_audit.json")
OUTPUT_MD = Path(__file__).with_name("sparsene_a800_n512_fp32_final.md")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def selected_tuning(config: dict[str, Any], dataset: str) -> dict[str, int]:
    tuning = dict(config["default"])
    tuning.update(config["datasets"].get(dataset, {}))
    return {
        "tile_n": int(tuning["tile_n"]),
        "max_tiles_per_execution_segment": int(
            tuning["max_tiles_per_execution_segment"]
        ),
    }


def collect(run_dir: Path, config: dict[str, Any]) -> list[dict[str, Any]]:
    run_datasets = read_json(run_dir / "run_manifest.json")["datasets"]
    if len(run_datasets) != 26 or len(set(run_datasets)) != 26:
        raise RuntimeError(f"expected 26 unique run datasets, got {len(run_datasets)}")
    if set(config["datasets"]) != set(run_datasets):
        missing = sorted(set(run_datasets) - set(config["datasets"]))
        extra = sorted(set(config["datasets"]) - set(run_datasets))
        raise RuntimeError(f"config/run dataset mismatch: missing={missing}, extra={extra}")

    rows: list[dict[str, Any]] = []
    for dataset in run_datasets:
        tuning = selected_tuning(config, dataset)
        variant = (
            f"n{config['n']}_tile{tuning['tile_n']}_segment"
            f"{tuning['max_tiles_per_execution_segment']}"
        )
        result_path = (
            run_dir
            / "datasets"
            / dataset
            / "operator_generation"
            / "e2e_results"
            / f"{variant}.json"
        )
        manifest_path = (
            run_dir
            / "datasets"
            / dataset
            / "operator_generation"
            / "generation_manifest.json"
        )
        result = read_json(result_path)
        manifest = read_json(manifest_path)
        plan = result["backend_plan"]
        if result["dataset"] != dataset or int(result["shape"]["N"]) != config["n"]:
            raise RuntimeError(f"{dataset}: identity/N mismatch in {result_path}")
        if result["timing"]["contract"] != config["performance_contract"]:
            raise RuntimeError(f"{dataset}: performance contract mismatch")
        if int(plan["tile_n"]) != tuning["tile_n"]:
            raise RuntimeError(f"{dataset}: result tile_n does not match final config")
        if (
            int(plan["max_tiles_per_execution_segment"])
            != tuning["max_tiles_per_execution_segment"]
        ):
            raise RuntimeError(f"{dataset}: result segment limit does not match final config")
        if not result["correctness"]["passed"]:
            raise RuntimeError(f"{dataset}: correctness failed; refusing to publish final table")
        launch = manifest.get("launch_contract", {})
        if launch.get("threads_per_block") != 32 or launch.get("synchronization") != "warp":
            raise RuntimeError(f"{dataset}: final one-warp launch contract is missing")

        timing = result["timing"]
        correctness = result["correctness"]
        packed = result["packed"]
        rows.append(
            {
                "dataset": dataset,
                **result["shape"],
                "backend": plan["backend"],
                "logical_tile": f"{plan['logical_blk_m']}x{plan['logical_blk_k']}",
                "physical_tile": f"{plan['physical_blk_m']}x{plan['physical_blk_k']}",
                "tile_n": plan["tile_n"],
                "max_tiles_per_execution_segment": plan[
                    "max_tiles_per_execution_segment"
                ],
                "row_reorder": plan["row_reorder"] or "none",
                "packing_mode": plan["packing_mode"],
                "row_windows": packed["row_windows"],
                "execution_segments": packed["execution_segments"],
                "generated_ms": timing["generated_ms"],
                "cusparse_ms": timing["cusparse_ms"],
                "speedup_vs_cusparse": timing["speedup_vs_cusparse"],
                "correctness": "PASS",
                "mismatches": correctness["mismatches"],
                "nonfinite": correctness["nonfinite"],
                "relative_l2_error": correctness["relative_l2_error"],
                "max_abs_normalized": correctness["max_abs_normalized"],
                "warmup": timing["warmup"],
                "repeat": timing["repeat"],
                "result_path": str(result_path.relative_to(REPO_ROOT)),
            }
        )
    return rows


def write_csv(rows: list[dict[str, Any]]) -> None:
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict[str, Any]], aggregate: dict[str, Any]) -> None:
    lines = [
        "# Sparsene generated operators: A800, N=512, FP32",
        "",
        "This is the correctness-closed final version selected by "
        "`a800_n512_fp32_final_config.json`. GPU benchmarks ran strictly serially.",
        "Format conversion, reorder/inverse mapping, and cuSPARSE preprocess are outside "
        "the timing contract; all generated compute/reduction launches are inside.",
        "",
        f"- Correctness: {aggregate['correctness_passed']}/26 PASS",
        f"- Faster than cuSPARSE: {aggregate['faster_than_cusparse']}/26",
        f"- Geometric-mean cuSPARSE/generated speedup: "
        f"{aggregate['geomean_speedup_vs_cusparse']:.4f}x",
        f"- Warmup/repeat: {aggregate['warmup']}/{aggregate['repeat']}",
        "",
        "| dataset | logical tile | tile N | segment | generated ms | cuSPARSE ms | speedup | correct |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['logical_tile']} | {row['tile_n']} | "
            f"{row['max_tiles_per_execution_segment']} | {row['generated_ms']:.6g} | "
            f"{row['cusparse_ms']:.6g} | {row['speedup_vs_cusparse']:.4f}x | PASS |"
        )
    lines.extend(
        [
            "",
            "Hardware note: these newly measured results are from an NVIDIA A800 80GB "
            "PCIe. Historical paper/SOTA tables in this directory are A100 results and "
            "must not be mixed into a same-hardware ranking.",
        ]
    )
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    config = read_json(args.config.resolve())
    if config["schema_version"] != "sparsene.custom_format_tuning.v1":
        raise RuntimeError("unsupported final tuning config")
    rows = collect(run_dir, config)
    speedups = [float(row["speedup_vs_cusparse"]) for row in rows]
    warmups = {int(row["warmup"]) for row in rows}
    repeats = {int(row["repeat"]) for row in rows}
    if len(warmups) != 1 or len(repeats) != 1:
        raise RuntimeError(f"mixed timing repetitions: warmup={warmups}, repeat={repeats}")
    aggregate = {
        "datasets": len(rows),
        "correctness_passed": sum(row["correctness"] == "PASS" for row in rows),
        "faster_than_cusparse": sum(value > 1.0 for value in speedups),
        "geomean_speedup_vs_cusparse": math.exp(
            sum(math.log(value) for value in speedups) / len(speedups)
        ),
        "warmup": warmups.pop(),
        "repeat": repeats.pop(),
    }
    write_csv(rows)
    write_markdown(rows, aggregate)
    OUTPUT_JSON.write_text(
        json.dumps(
            {
                "schema_version": "sparsene.a800_n512_fp32_final.v1",
                "run_dir": str(run_dir.relative_to(REPO_ROOT)),
                "config": str(args.config.resolve().relative_to(REPO_ROOT)),
                "performance_contract": config["performance_contract"],
                "execution_policy": config["execution_policy"],
                "aggregate": aggregate,
                "results": rows,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(aggregate, ensure_ascii=False))


if __name__ == "__main__":
    main()
