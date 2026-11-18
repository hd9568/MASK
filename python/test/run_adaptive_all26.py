#!/usr/bin/env python3
"""Run 26 joint-format datasets with CPU overlap and serialized GPU work."""

from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone


DATASETS = (
    "ASIC_680k", "DD", "Ga41As41H72", "OVCAR-8H", "Si41Ge41H72",
    "Yeast", "YeastH", "cant", "conf5_4-8x8-10", "consph", "cop20k_A",
    "dc2", "ddi", "eu-2005", "mac_econ_fwd500", "mc2depi", "mip1",
    "pdb1HYS", "protein", "pwtk", "reddit", "rma10", "scircuit",
    "shipsec1", "web-BerkStan", "webbase-1M",
)


def preflight(args: argparse.Namespace) -> None:
    if args.parallel < 1:
        raise SystemExit("--parallel must be positive")
    missing = [
        name for name in DATASETS
        if not (args.feature_dir / f"{name}.features.v3.json").is_file()
    ]
    if missing:
        raise SystemExit(f"Missing schema-v3 features: {', '.join(missing)}")
    if not args.ncu_bin.is_file():
        raise SystemExit(f"NCU binary missing: {args.ncu_bin}")
    import shutil
    if shutil.which("nvcc") is None:
        raise SystemExit("nvcc is not available on PATH")
    check = subprocess.run(["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if check.returncode or not any(("9.0" if args.platform == "h100" else "8.0") in line for line in check.stdout.splitlines()):
        raise SystemExit("No GPU matching requested platform: " + check.stderr.strip())
    from sparsene.agent.config import resolve_model_config

    model = resolve_model_config(
        model_name=args.model, yaml_path=str(args.model_config)
    )
    if (
        model.provider != "openai_compatible"
        or model.api_base_url != "https://api.openlux.ai/v1"
        or model.model != args.model
        or not model.api_key
        or not model.stream
        or (args.model == "deepseek-v4-pro" and model.thinking_mode != "enabled")
        or (args.model == "gpt-6-astra" and model.reasoning_effort not in {"low", "medium", "high", "xhigh", "max"})
    ):
        raise SystemExit("Model thinking/OpenLux configuration or OPENLUX_API_KEY is unavailable")
    from sparsene.utils.sparsene_reference import load_sparsene_reference
    for dataset in DATASETS:
        load_sparsene_reference(dataset)
    print(f"Preflight OK: {len(DATASETS)} datasets; workers={min(args.parallel, len(DATASETS))}; format0=A100 historical best format; formats<=6; strategy<=iter0..iter5; GPU locked", flush=True)


def probe(args: argparse.Namespace) -> bool:
    """Exercise the same streaming OpenLux route used by the format Agent."""
    from sparsene.agent.client import call_openai_compatible
    from sparsene.agent.config import resolve_model_config

    config = resolve_model_config(
        model_name=args.model, yaml_path=str(args.model_config)
    )
    directory = args.run_dir / "connectivity"
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    path = directory / f"{args.model}_openlux_probe_{stamp}.json"
    record = {
        "model": config.to_dict(),
        "request": {
            "stream": config.stream,
            "provider_sort": config.provider_sort,
            "response_format": {"type": "json_object"},
            "max_tokens": 1024,
            "temperature": config.temperature,
        },
        "status": "failed",
    }
    try:
        response = call_openai_compatible(
            model=config.model,
            api_key=config.api_key,
            api_base_url=config.api_base_url,
            system_prompt="Return one small JSON object.",
            user_prompt='Return exactly {"ok":true}.',
            temperature=config.temperature,
            max_tokens=1024,
            timeout_sec=min(config.timeout_sec, 90),
            stream=config.stream,
            provider_sort=config.provider_sort,
            thinking_mode=config.thinking_mode, reasoning_effort=config.reasoning_effort,
            response_format={"type": "json_object"},
        )
        payload = json.loads(response.content)
        if payload.get("ok") is not True or response.finish_reason == "length":
            raise ValueError("probe did not return complete {ok:true} JSON")
        record.update(status="ok", elapsed_s=response.elapsed_s,
                      usage=response.usage.to_dict(), finish_reason=response.finish_reason)
    except Exception as exc:
        record.update(error_type=type(exc).__name__,
                      error=str(exc).replace(config.api_key, "[REDACTED]")[:500])
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"OpenLux connectivity: {record['status']}; audit: {path}", flush=True)
    return record["status"] == "ok"


def worker_command(args: argparse.Namespace, dataset: str) -> list[str]:
    return [
        sys.executable, "-u", str(args.repo / "python/test/run_joint_format_strategy_feedback.py"),
        "--run-dir", str(args.run_dir / "workers" / dataset),
        "--feature-dir", str(args.feature_dir),
        "--datasets", dataset,
        "--format-iterations", "6",
        "--initial-format", "sparsene_historical_best",
        *(["--historical-format-only"] if args.platform == "h100" else []),
        "--strategy-agent-iterations", "5",
        "--incumbent-dir", str(args.incumbent_dir), "--skip-incumbent-update",
        "--format-model", args.model,
        "--format-max-tokens", "32768",
        "--format-timeout-sec", "900",
        "--format-max-attempts", "2",
        "--format-retry-delay-sec", "1",
        "--strategy-provider", "openai_http",
        "--strategy-model", args.model,
        "--strategy-max-tokens", "32768",
        "--strategy-timeout-sec", "600",
        "--model-config", str(args.model_config),
        "--n", "512", "--warmup", "3", "--repeat", "32",
        "--measurement-runs", "3", "--ncu-bin", str(args.ncu_bin),
        "--resume",
    ]


def run_one(args: argparse.Namespace, dataset: str) -> tuple[str, int]:
    work = args.run_dir / "workers" / dataset
    work.mkdir(parents=True, exist_ok=True)
    command = worker_command(args, dataset)
    with (work / "driver.log").open("a", encoding="utf-8") as log:
        log.write("\nCOMMAND: " + " ".join(command) + "\n")
        log.flush()
        with subprocess.Popen(
            command, cwd=args.repo, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
        ) as process:
            assert process.stdout is not None
            for line in process.stdout:
                log.write(line)
                log.flush()
                if line.startswith(("[01/01]", "  format ", "complete ", "format ")) or any(
                    marker in line for marker in ("Traceback", "ERROR", "failed after")
                ):
                    print(f"[{dataset}] {line.rstrip()}", flush=True)
            return dataset, process.wait()


def summarize(args: argparse.Namespace) -> None:
    args.run_dir.mkdir(parents=True, exist_ok=True)
    fields = (
        "dataset", "driver_returncode", "status", "formats_recorded",
        "historical_sparsene_variant", "historical_sparsene_ms",
        "generated_iteration_0_vs_historical_sparsene",
        "strategy_evaluated", "strategy_validated", "strategy_failed",
        "best_current_run_ms", "best_current_run_speedup_vs_cusparse",
        "best_current_run_vs_historical_sparsene",
        "best_current_format", "best_current_strategy", "best_current_operator",
    )
    rows = []
    for dataset in DATASETS:
        work = args.run_dir / "workers" / dataset
        summary_path = work / "datasets" / dataset / "joint_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
        counts = summary.get("counts") or {}
        initial_record = next(
            (record for record in summary.get("format_iterations", [])
             if record.get("format_iteration") == 0), {}
        )
        returncode_path = work / "driver_returncode.json"
        returncode = json.loads(returncode_path.read_text(encoding="utf-8")).get("returncode") if returncode_path.is_file() else None
        rows.append({
            "dataset": dataset,
            "driver_returncode": returncode,
            "status": summary.get("status", "missing"),
            "historical_sparsene_variant": summary.get("historical_sparsene_variant"),
            "historical_sparsene_ms": summary.get("historical_sparsene_ms"),
            "generated_iteration_0_vs_historical_sparsene": initial_record.get(
                "generated_iteration_0_vs_historical_sparsene"
            ),
            "formats_recorded": counts.get("format_iterations_recorded"),
            "strategy_evaluated": counts.get("strategy_candidates_evaluated"),
            "strategy_validated": counts.get("strategy_candidates_validated"),
            "strategy_failed": counts.get("strategy_candidates_failed"),
            "best_current_run_ms": summary.get("best_current_run_generated_ms"),
            "best_current_run_speedup_vs_cusparse": summary.get("best_current_run_speedup_vs_cusparse"),
            "best_current_run_vs_historical_sparsene": summary.get("best_current_run_vs_historical_sparsene"),
            "best_current_format": summary.get("best_current_run_format_iteration"),
            "best_current_strategy": summary.get("best_current_run_strategy_iteration"),
            "best_current_operator": summary.get("best_current_run_operator_path"),
        })
    output = args.run_dir / "campaign_summary.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Summary: {output}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "probe", "run", "summarize"))
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--seed-run-dir", type=Path, required=True)
    parser.add_argument("--incumbent-dir", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--ncu-bin", type=Path, required=True)
    parser.add_argument("--parallel", type=int, default=8)
    parser.add_argument("--model", choices=("deepseek-v4-pro", "gpt-6-astra"), required=True)
    parser.add_argument("--platform", choices=("a800", "h100"), required=True)
    args = parser.parse_args()
    if args.action == "summarize":
        summarize(args)
        return 0
    preflight(args)
    if args.action == "check":
        return 0
    if args.action == "probe":
        return 0 if probe(args) else 1
    args.run_dir.mkdir(parents=True, exist_ok=True)
    failed = 0
    with ThreadPoolExecutor(max_workers=min(args.parallel, len(DATASETS))) as pool:
        futures = {pool.submit(run_one, args, dataset): dataset for dataset in DATASETS}
        for future in as_completed(futures):
            dataset = futures[future]
            try:
                _, returncode = future.result()
            except Exception as exc:
                returncode = 1
                print(f"[{dataset}] worker exception: {type(exc).__name__}: {exc}", flush=True)
            work = args.run_dir / "workers" / dataset
            work.mkdir(parents=True, exist_ok=True)
            (work / "driver_returncode.json").write_text(
                json.dumps({"dataset": dataset, "returncode": returncode}, indent=2) + "\n",
                encoding="utf-8",
            )
            failed += returncode != 0
            print(f"[{dataset}] worker exit={returncode}", flush=True)
            summarize(args)
    print(f"Campaign finished: datasets={len(DATASETS)} worker_failures={failed}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
