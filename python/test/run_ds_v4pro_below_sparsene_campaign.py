#!/usr/bin/env python3
"""Select below-Sparsene DeepSeek V4 Pro cases and run independent joint loops."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


BASELINE = "sparsene1_paper_speedup_vs_cusparse"
DEEPSEEK = "sparsekg_ds_v4pro_speedup"


def select(table: Path) -> list[dict[str, str]]:
    with table.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not {"dataset", BASELINE, DEEPSEEK} <= rows[0].keys():
        raise SystemExit(f"Missing comparison columns in {table}")
    selected = [row for row in rows if float(row[DEEPSEEK]) < float(row[BASELINE])]
    if not selected:
        raise SystemExit(f"No DeepSeek V4 Pro cases below Sparsene in {table}")
    names = [row["dataset"] for row in selected]
    if len(names) != len(set(names)):
        raise SystemExit("Comparison table has duplicate selected datasets")
    return selected


def frozen_targets(args: argparse.Namespace, *, create: bool) -> list[dict[str, str]]:
    path = args.run_dir / "targets.csv"
    if path.exists():
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    targets = select(args.table)
    if create:
        args.run_dir.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=("dataset", BASELINE, DEEPSEEK))
            writer.writeheader()
            writer.writerows({key: row[key] for key in writer.fieldnames} for row in targets)
    return targets


def preflight(args: argparse.Namespace, targets: list[dict[str, str]]) -> None:
    if args.parallel < 1:
        raise SystemExit("--parallel must be positive")
    if not args.seed_run_dir.is_dir():
        raise SystemExit(f"Seed run missing: {args.seed_run_dir}")
    if not (args.incumbent_dir / "summary.csv").is_file():
        raise SystemExit(f"Incumbent registry missing: {args.incumbent_dir}")
    if not args.ncu_bin.is_file():
        raise SystemExit(f"NCU missing: {args.ncu_bin}")
    missing = [row["dataset"] for row in targets if not (args.feature_dir / f"{row['dataset']}.features.v3.json").is_file()]
    if missing:
        raise SystemExit(f"Missing feature files: {', '.join(missing)}")
    from sparsene.agent.config import resolve_model_config

    model = resolve_model_config(model_name="deepseek-v4-pro", yaml_path=str(args.model_config))
    if model.provider != "openai_compatible" or model.api_base_url != "https://api.openlux.ai/v1" or not model.api_key or not model.stream or model.provider_sort != "success_rate":
        raise SystemExit("OPENLUX_API_KEY or DeepSeek V4 Pro OpenLux model configuration is invalid")
    if args.initial_format == "sr_bcrs_16x8":
        from run_joint_format_strategy_feedback import _fixed_sr_bcrs_response

        _fixed_sr_bcrs_response()
    elif args.initial_format == "sparsene_historical_best":
        from run_joint_format_strategy_feedback import _fixed_sr_bcrs_response
        from sparsene.utils.sparsene_reference import load_sparsene_reference

        for row in targets:
            _fixed_sr_bcrs_response(load_sparsene_reference(row["dataset"]))
    print(f"Selected {len(targets)} below-Sparsene datasets; CPU cores={os.cpu_count()}, dataset workers={min(args.parallel, len(targets))}.", flush=True)
    print(f"Format 0: {args.initial_format}; formats={args.format_iterations}; strategy iterations=0..{args.strategy_agent_iterations} per format.", flush=True)
    print("GPU validation, timing and NCU use the shared cross-process lock; this check makes no API call.", flush=True)
    for row in targets:
        ratio = float(row[DEEPSEEK]) / float(row[BASELINE])
        print(f"  {row['dataset']:<18} Sparsene={float(row[BASELINE]):.4f} DeepSeek={float(row[DEEPSEEK]):.4f} DS/Sparsene={ratio:.4f}", flush=True)


def run_one(args: argparse.Namespace, dataset: str) -> tuple[str, int]:
    work = args.run_dir / "workers" / dataset
    work.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, "-u", str(args.repo / "python/test/run_joint_format_strategy_feedback.py"),
        "--run-dir", str(work), "--feature-dir", str(args.feature_dir),
        "--datasets", dataset, "--format-iterations", str(args.format_iterations),
        "--initial-format", args.initial_format,
        "--strategy-agent-iterations", str(args.strategy_agent_iterations),
        "--initial-feedback-run-dir", str(args.seed_run_dir), "--incumbent-dir", str(args.incumbent_dir),
        "--format-model", "deepseek-v4-pro", "--format-max-tokens", "32768",
        "--format-timeout-sec", "900", "--format-max-attempts", "3", "--format-retry-delay-sec", "5",
        "--strategy-provider", "openai_http", "--strategy-model", "deepseek-v4-pro",
        "--strategy-max-tokens", "16384", "--strategy-timeout-sec", "600",
        "--model-config", str(args.model_config), "--n", "512", "--warmup", "3",
        "--repeat", "32", "--measurement-runs", "3", "--ncu-bin", str(args.ncu_bin),
        "--abort-on-format-failure", "--resume",
    ]
    with (work / "driver.log").open("a", encoding="utf-8") as log:
        log.write("\nCOMMAND: " + " ".join(command) + "\n")
        log.flush()
        result = subprocess.run(command, cwd=args.repo, stdout=log, stderr=subprocess.STDOUT, check=False)
    return dataset, result.returncode


def summarize(args: argparse.Namespace, targets: list[dict[str, str]]) -> None:
    if not args.run_dir.is_dir():
        raise SystemExit(f"Run directory missing: {args.run_dir}")
    fields = ["dataset", "sparsene_speedup", "deepseek_previous_speedup", "deepseek_previous_vs_sparsene",
              "new_best_current_run_speedup", "new_vs_sparsene", "new_vs_previous", "status",
              "best_format_iteration", "best_strategy_iteration", "best_generated_ms", "best_operator_path"]
    output = args.run_dir / "per_dataset_summary.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in targets:
            dataset = row["dataset"]
            path = args.run_dir / "workers" / dataset / "datasets" / dataset / "joint_summary.json"
            summary = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
            new = summary.get("best_current_run_speedup_vs_cusparse")
            old, baseline = float(row[DEEPSEEK]), float(row[BASELINE])
            writer.writerow({
                "dataset": dataset, "sparsene_speedup": baseline, "deepseek_previous_speedup": old,
                "deepseek_previous_vs_sparsene": old / baseline,
                "new_best_current_run_speedup": new, "new_vs_sparsene": float(new) / baseline if new else "",
                "new_vs_previous": float(new) / old if new else "", "status": summary.get("status", "not_started"),
                "best_format_iteration": summary.get("best_current_run_format_iteration"),
                "best_strategy_iteration": summary.get("best_current_run_strategy_iteration"),
                "best_generated_ms": summary.get("best_current_run_generated_ms"),
                "best_operator_path": summary.get("best_current_run_operator_path"),
            })
    print(f"Summary: {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "run", "summarize"))
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--seed-run-dir", type=Path, required=True)
    parser.add_argument("--incumbent-dir", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--ncu-bin", type=Path, required=True)
    parser.add_argument("--parallel", type=int, required=True)
    parser.add_argument("--format-iterations", type=int, default=3)
    parser.add_argument("--initial-format", choices=("agent", "sr_bcrs_16x8", "sparsene_historical_best"), default="agent")
    parser.add_argument("--strategy-agent-iterations", type=int, default=5)
    args = parser.parse_args()
    targets = frozen_targets(args, create=args.action == "run")
    if args.action == "summarize":
        summarize(args, targets)
        return
    preflight(args, targets)
    if args.action == "check":
        return
    failures = 0
    with ThreadPoolExecutor(max_workers=min(args.parallel, len(targets))) as pool:
        futures = {pool.submit(run_one, args, row["dataset"]): row["dataset"] for row in targets}
        for future in as_completed(futures):
            dataset = futures[future]
            try:
                _, code = future.result()
            except Exception as exc:
                code = 1
                print(f"[{dataset}] worker exception: {exc}", flush=True)
            print(f"[{dataset}] {'COMPLETED' if code == 0 else 'FAILED'} exit={code} log={args.run_dir / 'workers' / dataset / 'driver.log'}", flush=True)
            failures += code != 0
            summarize(args, targets)
    print(f"Campaign finished: {len(targets) - failures}/{len(targets)} workers completed, {failures} failed.", flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
