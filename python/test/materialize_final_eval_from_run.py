from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Dict, List

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from run_strategy_feedback_multidataset import (
    DEFAULT_INC_PATH,
    _aggregate_dataset_results,
    _build_binary,
    _discover_datasets,
    _run_dataset,
    _write_best_table,
    _write_dataset_csv,
    _write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--iteration", type=int, required=True, help="Use 0 for baseline_heuristic, otherwise N for iter_NN.")
    parser.add_argument("--dataset-root", type=str, default=None)
    parser.add_argument("--dataset-glob", type=str, default=None)
    parser.add_argument("--max-datasets", type=int, default=None)
    parser.add_argument("--final-warmup", type=int, default=None)
    parser.add_argument("--final-repeat", type=int, default=None)
    parser.add_argument("--timeout-sec", type=int, default=None)
    return parser.parse_args()


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _iteration_dir(run_dir: Path, iteration: int) -> Path:
    if iteration == 0:
        return run_dir / "baseline_heuristic"
    return run_dir / f"iter_{iteration:02d}"


def _load_run_args(run_dir: Path, overrides: argparse.Namespace) -> argparse.Namespace:
    run_config = _load_json(run_dir / "run_config.json")
    if overrides.dataset_root is not None:
        run_config["dataset_root"] = overrides.dataset_root
    if overrides.dataset_glob is not None:
        run_config["dataset_glob"] = overrides.dataset_glob
    if overrides.max_datasets is not None:
        run_config["max_datasets"] = overrides.max_datasets
    if overrides.final_warmup is not None:
        run_config["final_warmup"] = overrides.final_warmup
    if overrides.final_repeat is not None:
        run_config["final_repeat"] = overrides.final_repeat
    if overrides.timeout_sec is not None:
        run_config["timeout_sec"] = overrides.timeout_sec
    return argparse.Namespace(**run_config)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    run_args = _load_run_args(run_dir, args)
    target_dir = _iteration_dir(run_dir, args.iteration)
    kernel_path = target_dir / "me_tcf_kernel3.inc"
    if not kernel_path.exists():
        raise FileNotFoundError(f"Kernel file not found: {kernel_path}")

    datasets = _discover_datasets(run_args)
    if not datasets:
        raise RuntimeError(f"No datasets found under {run_args.dataset_root}")

    final_dir = run_dir / f"manual_final_eval_iter_{args.iteration:02d}"
    final_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        final_dir / "recovery_config.json",
        {
            "run_dir": str(run_dir),
            "iteration": args.iteration,
            "dataset_root": run_args.dataset_root,
            "dataset_glob": run_args.dataset_glob,
            "max_datasets": run_args.max_datasets,
            "final_warmup": run_args.final_warmup,
            "final_repeat": run_args.final_repeat,
            "timeout_sec": run_args.timeout_sec,
        },
    )

    restore_needed = DEFAULT_INC_PATH.exists()
    backup_path = final_dir / "pre_recovery_me_tcf_kernel3.inc"
    if restore_needed:
        shutil.copy2(DEFAULT_INC_PATH, backup_path)
    shutil.copy2(kernel_path, DEFAULT_INC_PATH)

    try:
        binary_path = _build_binary(run_args, final_dir)
        dataset_log_dir = final_dir / "dataset_logs"
        dataset_log_dir.mkdir(parents=True, exist_ok=True)
        results: List[Dict[str, Any]] = [
            _run_dataset(
                binary_path,
                dataset_path,
                n_cols=run_args.n_cols,
                warmup=run_args.final_warmup,
                repeat=run_args.final_repeat,
                timeout_sec=run_args.timeout_sec,
                iter_log_dir=dataset_log_dir,
            )
            for dataset_path in datasets
        ]
        aggregate = _aggregate_dataset_results(results)
        _write_dataset_csv(final_dir / "dataset_results.csv", results)
        _write_json(final_dir / "aggregate.json", aggregate)
    finally:
        if restore_needed and backup_path.exists():
            shutil.copy2(backup_path, DEFAULT_INC_PATH)
        elif not restore_needed and DEFAULT_INC_PATH.exists():
            DEFAULT_INC_PATH.unlink()

    csv_path, md_path = _write_best_table(
        rows=results,
        aggregate=aggregate,
        iteration_label=str(args.iteration),
        run_dir=run_dir,
        prefix="best_iteration_all_datasets_final",
    )
    _write_json(run_dir / "best_iteration_final_aggregate.json", aggregate)
    shutil.copy2(kernel_path, run_dir / "best_iteration_kernel.inc")
    print(f"Recovered final evaluation for iteration {args.iteration}")
    print(f"CSV: {csv_path}")
    print(f"Markdown: {md_path}")
    print(f"Aggregate: {run_dir / 'best_iteration_final_aggregate.json'}")


if __name__ == "__main__":
    main()
