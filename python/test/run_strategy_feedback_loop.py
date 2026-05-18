from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any, Dict, List, Optional

from sparsene.formats.DTC_SpMM import ME_TCF_FORMAT
from sparsene.logging import get_logger, set_logging_level_for_all
from sparsene.op_gen.computent.computent import computent_from_rts
from sparsene.op_gen.nvir.generate import SPARSE_FORMAT_REGISTRY, build_gmem_interfaces
from sparsene.op_gen.opir.cValFlattenPass import CValFlattenPass
from sparsene.op_gen.opir.generate import generate_from_computent
from sparsene.op_gen.opir.varlenLoweringPass import VarlenLoweringPass
from sparsene.op_gen.strategy_agent import (
    StrategyConfig,
    StrategyFeedbackContext,
    run_strategy_agent,
)
from sparsene.transform.rts import derive_rts


logger = get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
PY_DIR = REPO_ROOT / "python"
TESTBED_DIR = REPO_ROOT / "examples" / "src_fp32" / "dtc" / "testbed"
DEFAULT_INC_PATH = PY_DIR / "me_tcf_kernel3.inc"
DEFAULT_LOOP_LOG_DIR = REPO_ROOT / "docs" / "agent_feedback_runs"
DEFAULT_BUILD_BINARY = TESTBED_DIR / "build_generated" / "demo-dtc-generated-tf32"
DEFAULT_NCU_BIN = Path.home() / "local" / "cuda" / "bin" / "ncu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", "-d", action="store_true")
    parser.add_argument(
        "--strategy-provider",
        type=str,
        default="openai_http",
        choices=[
            "heuristic",
            "manual_json",
            "openai_http",
            "anthropic_http",
            "subprocess_json",
        ],
    )
    parser.add_argument("--strategy-model", type=str, default=None)
    parser.add_argument("--strategy-api-key", type=str, default=None)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--seed-heuristic-baseline", action="store_true", default=True)
    parser.add_argument("--no-seed-heuristic-baseline", action="store_false", dest="seed_heuristic_baseline")
    parser.add_argument("--loop-log-dir", type=str, default=str(DEFAULT_LOOP_LOG_DIR))
    parser.add_argument("--perf-warmup", type=int, default=5)
    parser.add_argument("--perf-repeat", type=int, default=20)
    parser.add_argument("--validate-warmup", type=int, default=0)
    parser.add_argument("--validate-repeat", type=int, default=1)
    parser.add_argument("--profile-warmup", type=int, default=1)
    parser.add_argument("--profile-repeat", type=int, default=3)
    parser.add_argument("--ncu-bin", type=str, default=str(DEFAULT_NCU_BIN))
    parser.add_argument("--include-full-inc-code", action="store_true", default=False)
    parser.add_argument("--no-include-full-inc-code", action="store_false", dest="include_full_inc_code")
    parser.add_argument("--strategy-timeout-sec", type=int, default=180)
    return parser.parse_args()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def _run_command(
    command: List[str],
    *,
    cwd: Path,
    env_overrides: Optional[Dict[str, str]] = None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)
    return subprocess.run(
        command,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def _clean_num(value: str) -> Optional[float]:
    stripped = value.strip().replace(",", "")
    if stripped in {"", "nan", "NaN", "None"}:
        return None
    try:
        return float(stripped)
    except ValueError:
        return None


def _parse_perf_output(text: str) -> Dict[str, Any]:
    my_match = re.search(r"mykernel_time:\s*([0-9.]+)\s*ms", text)
    cu_match = re.search(r"cusparse_time:\s*([0-9.]+)\s*ms", text)
    result: Dict[str, Any] = {
        "mykernel_ms": float(my_match.group(1)) if my_match else None,
        "cusparse_ms": float(cu_match.group(1)) if cu_match else None,
    }
    my_ms = result["mykernel_ms"]
    cu_ms = result["cusparse_ms"]
    if my_ms and cu_ms:
        result["ratio_vs_cusparse"] = my_ms / cu_ms if cu_ms else None
        result["speedup_vs_generated"] = cu_ms / my_ms if my_ms else None
    else:
        result["ratio_vs_cusparse"] = None
        result["speedup_vs_generated"] = None
    return result


def _parse_validation_output(text: str, returncode: int) -> Dict[str, Any]:
    error_count = len(re.findall(r"Error\(", text))
    nan_match = re.search(r"check nan:\s*cusp\s*([0-9.\-]+)\s*mykernel\s*([0-9.\-]+)", text)
    passed = returncode == 0 and error_count == 0 and nan_match is not None
    return {
        "passed": passed,
        "returncode": returncode,
        "error_count": error_count,
        "nan_check_cusparse": float(nan_match.group(1)) if nan_match else None,
        "nan_check_mykernel": float(nan_match.group(2)) if nan_match else None,
    }


def _parse_ncu_csv(text: str) -> Dict[str, Any]:
    metric_aliases = {
        ("Launch Statistics", "Block Size"): "block_size",
        ("Launch Statistics", "Grid Size"): "grid_size",
        ("Launch Statistics", "Registers Per Thread"): "registers_per_thread",
        ("Launch Statistics", "Static Shared Memory Per Block"): "static_smem_per_block",
        ("Launch Statistics", "Waves Per SM"): "waves_per_sm",
        ("GPU Speed Of Light Throughput", "Memory Throughput"): "memory_throughput_pct",
        ("GPU Speed Of Light Throughput", "DRAM Throughput"): "dram_throughput_pct",
        ("GPU Speed Of Light Throughput", "Compute (SM) Throughput"): "compute_throughput_pct",
        ("Memory Workload Analysis", "L1/TEX Hit Rate"): "l1tex_hit_rate_pct",
        ("Memory Workload Analysis", "L2 Hit Rate"): "l2_hit_rate_pct",
        ("Memory Workload Analysis", "Mem Pipes Busy"): "mem_pipes_busy_pct",
    }
    collected: Dict[str, List[float]] = {}
    string_metrics: Dict[str, str] = {}
    for row in csv.reader(io.StringIO(text)):
        if len(row) < 15:
            continue
        section = row[11].strip()
        metric = row[12].strip()
        value = row[14].strip()
        alias = metric_aliases.get((section, metric))
        if alias is None:
            continue
        numeric_value = _clean_num(value)
        if numeric_value is not None:
            collected.setdefault(alias, []).append(numeric_value)
        elif value:
            string_metrics[alias] = value

    summary: Dict[str, Any] = {}
    for key, values in collected.items():
        summary[key] = round(sum(values) / len(values), 4)
    summary.update(string_metrics)

    advice = []
    for line in text.splitlines():
        if "SOLBottleneck" in line or "This kernel grid is too small" in line:
            advice.append(line.strip())
    summary["advice"] = advice[:3]
    return summary


def _extract_ncu_metrics_csv(text: str) -> str:
    csv_lines: List[str] = []
    in_csv = False
    for line in text.splitlines():
        if not in_csv and line.startswith('"ID"'):
            in_csv = True
        if in_csv:
            if '"GPU Speed Of Light Throughput","Duration",' in line:
                continue
            csv_lines.append(line)
    return "\n".join(csv_lines) + ("\n" if csv_lines else "")


def _build_flattened_ops():
    rts = derive_rts(ME_TCF_FORMAT)
    computent = computent_from_rts("ME_TCF", rts)
    ops = generate_from_computent(computent)
    lowered_ops = VarlenLoweringPass(
        op_builder=None,
        varlen2LenArrayTable=computent.varlen2LenArrayTable,
    ).run(ops)
    flattened_ops = CValFlattenPass(op_builder=None).run(lowered_ops)
    return flattened_ops


def _generate_inc_for_strategy(
    *,
    decision,
    iter_dir: Path,
) -> Path:
    strategy_json_path = iter_dir / "strategy_for_codegen.json"
    strategy_json_path.write_text(json.dumps(decision.to_dict(), indent=2, ensure_ascii=False))
    result = _run_command(
        [
            "python",
            "test/test_generate.py",
            "--strategy-provider",
            "manual_json",
            "--strategy-manual-json",
            str(strategy_json_path),
            "--strategy-log-dir",
            str(iter_dir / "codegen_agent_logs"),
        ],
        cwd=PY_DIR,
    )
    (iter_dir / "codegen.log").write_text(result.stdout)
    if result.returncode != 0:
        raise RuntimeError(
            f"manual_json codegen failed for {strategy_json_path}; see {iter_dir / 'codegen.log'}"
        )
    return DEFAULT_INC_PATH


def _build_gmem_inouts():
    flattened_ops = _build_flattened_ops()
    return build_gmem_interfaces(flattened_ops, SPARSE_FORMAT_REGISTRY["ME_TCF"])


def _strategy_brief(decision) -> Dict[str, Any]:
    return decision.to_dict()


def _build_feedback_notes(
    *,
    previous_entry: Optional[Dict[str, Any]],
    best_entry: Optional[Dict[str, Any]],
) -> List[str]:
    notes: List[str] = []
    if previous_entry:
        prev_perf = previous_entry.get("performance", {})
        notes.append(
            "Last iteration mykernel_ms={mykernel_ms}, ratio_vs_cusparse={ratio_vs_cusparse}".format(
                mykernel_ms=prev_perf.get("mykernel_ms"),
                ratio_vs_cusparse=prev_perf.get("ratio_vs_cusparse"),
            )
        )
    if best_entry:
        best_perf = best_entry.get("performance", {})
        notes.append(
            "Best correct iteration so far is iter={iteration} with mykernel_ms={mykernel_ms}".format(
                iteration=best_entry.get("iteration"),
                mykernel_ms=best_perf.get("mykernel_ms"),
            )
        )
    return notes


def _request_strategy_with_retry(
    *,
    format_name: str,
    gmem_inouts,
    strategy_config: StrategyConfig,
    feedback_context: StrategyFeedbackContext,
    iter_dir: Path,
):
    try:
        return run_strategy_agent(
            format_name=format_name,
            gmem_inouts=gmem_inouts,
            config=strategy_config,
            feedback_context=feedback_context,
        )
    except Exception as exc:
        if not feedback_context.include_full_kernel_code:
            raise
        fallback_context = StrategyFeedbackContext(**feedback_context.to_dict())
        fallback_context.include_full_kernel_code = False
        fallback_context.additional_notes = list(fallback_context.additional_notes) + [
            "Full .inc kernel code was omitted on retry because the previous API call timed out or failed.",
            f"Retry reason: {exc}",
        ]
        (iter_dir / "agent_retry_note.txt").write_text(
            "Retrying strategy request without full .inc kernel code.\n"
            f"Reason: {exc}\n"
        )
        return run_strategy_agent(
            format_name=format_name,
            gmem_inouts=gmem_inouts,
            config=strategy_config,
            feedback_context=fallback_context,
        )


def _run_validation(iter_dir: Path, args: argparse.Namespace) -> Dict[str, Any]:
    if DEFAULT_BUILD_BINARY.exists():
        DEFAULT_BUILD_BINARY.unlink()
    result = _run_command(
        [
            "./scripts/validate_generated_kernel.sh",
            "-warmup",
            str(args.validate_warmup),
            "-repeat",
            str(args.validate_repeat),
        ],
        cwd=TESTBED_DIR,
        env_overrides={
            "FORCE_RUN": "1",
            "PATCHED_ONLY": "0",
            "SKIP_GENERATE": "1",
        },
    )
    (iter_dir / "validate.log").write_text(result.stdout)
    parsed = _parse_validation_output(result.stdout, result.returncode)
    parsed["build_succeeded"] = result.returncode == 0 and DEFAULT_BUILD_BINARY.exists()
    parsed["raw_log"] = str(iter_dir / "validate.log")
    return parsed


def _run_performance(iter_dir: Path, args: argparse.Namespace) -> Dict[str, Any]:
    if not DEFAULT_BUILD_BINARY.exists():
        return {
            "mykernel_ms": None,
            "cusparse_ms": None,
            "ratio_vs_cusparse": None,
            "speedup_vs_generated": None,
            "returncode": None,
            "raw_log": None,
        }
    result = _run_command(
        [
            str(DEFAULT_BUILD_BINARY),
            "-warmup",
            str(args.perf_warmup),
            "-repeat",
            str(args.perf_repeat),
        ],
        cwd=TESTBED_DIR,
    )
    (iter_dir / "perf.log").write_text(result.stdout)
    parsed = _parse_perf_output(result.stdout)
    parsed["returncode"] = result.returncode
    parsed["raw_log"] = str(iter_dir / "perf.log")
    return parsed


def _run_ncu_profile(iter_dir: Path, args: argparse.Namespace) -> Dict[str, Any]:
    ncu_bin = Path(args.ncu_bin)
    if not ncu_bin.exists() or not DEFAULT_BUILD_BINARY.exists():
        return {"available": False}
    result = _run_command(
        [
            str(ncu_bin),
            "--target-processes",
            "all",
            "--kernel-name-base",
            "demangled",
            "--kernel-name",
            "regex:generated_kernel",
            "--section",
            "LaunchStats",
            "--section",
            "MemoryWorkloadAnalysis",
            "--section",
            "SpeedOfLight",
            "--csv",
            str(DEFAULT_BUILD_BINARY),
            "-warmup",
            str(args.profile_warmup),
            "-repeat",
            str(args.profile_repeat),
        ],
        cwd=TESTBED_DIR,
    )
    metrics_csv = _extract_ncu_metrics_csv(result.stdout)
    (iter_dir / "ncu_metrics.csv").write_text(metrics_csv)
    summary = _parse_ncu_csv(metrics_csv)
    summary["available"] = result.returncode == 0
    summary["returncode"] = result.returncode
    summary["raw_log"] = str(iter_dir / "ncu_metrics.csv")
    return summary


def _select_best_iteration(history: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    correct_entries = [
        item
        for item in history
        if item.get("validation", {}).get("passed")
        and item.get("performance", {}).get("mykernel_ms") is not None
    ]
    if not correct_entries:
        return None
    return min(correct_entries, key=lambda item: item["performance"]["mykernel_ms"])


def _write_curve_csv(run_dir: Path, history: List[Dict[str, Any]]) -> Path:
    curve_path = run_dir / "performance_curve.csv"
    with curve_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "iteration",
                "passed",
                "mykernel_ms",
                "cusparse_ms",
                "ratio_vs_cusparse",
                "registers_per_thread",
                "waves_per_sm",
                "memory_throughput_pct",
                "compute_throughput_pct",
            ]
        )
        for item in history:
            writer.writerow(
                [
                    item.get("iteration"),
                    item.get("validation", {}).get("passed"),
                    item.get("performance", {}).get("mykernel_ms"),
                    item.get("performance", {}).get("cusparse_ms"),
                    item.get("performance", {}).get("ratio_vs_cusparse"),
                    item.get("profile_summary", {}).get("registers_per_thread"),
                    item.get("profile_summary", {}).get("waves_per_sm"),
                    item.get("profile_summary", {}).get("memory_throughput_pct"),
                    item.get("profile_summary", {}).get("compute_throughput_pct"),
                ]
            )
    return curve_path


def _write_curve_plot(run_dir: Path, history: List[Dict[str, Any]]) -> Path:
    plot_path = run_dir / "performance_curve.svg"
    entries = [item for item in history if isinstance(item.get("iteration"), int) and item.get("iteration") > 0]
    kernel_points = [
        (item["iteration"], item.get("performance", {}).get("mykernel_ms"))
        for item in entries
        if item.get("performance", {}).get("mykernel_ms") is not None
    ]
    cusparse_values = [
        item.get("performance", {}).get("cusparse_ms")
        for item in entries
        if item.get("performance", {}).get("cusparse_ms") is not None
    ]
    if not kernel_points or not cusparse_values:
        plot_path.write_text(
            "<svg xmlns='http://www.w3.org/2000/svg' width='900' height='420'>"
            "<text x='20' y='40' font-size='20'>No valid performance data.</text></svg>"
        )
        return plot_path

    cusparse_ms = sum(cusparse_values) / len(cusparse_values)
    y_values = [point[1] for point in kernel_points] + [cusparse_ms]
    y_min = min(y_values)
    y_max = max(y_values)
    if math.isclose(y_min, y_max):
        y_min *= 0.9
        y_max *= 1.1
    padding = (y_max - y_min) * 0.1
    y_min -= padding
    y_max += padding

    width = 960
    height = 520
    left = 90
    right = 30
    top = 40
    bottom = 80
    plot_w = width - left - right
    plot_h = height - top - bottom

    def x_pos(iteration: int) -> float:
        if len(entries) == 1:
            return left + plot_w / 2
        return left + (iteration - 1) * plot_w / max(len(entries) - 1, 1)

    def y_pos(value: float) -> float:
        if math.isclose(y_max, y_min):
            return top + plot_h / 2
        return top + (y_max - value) * plot_h / (y_max - y_min)

    grid_lines = []
    tick_count = 5
    for i in range(tick_count + 1):
        y_value = y_min + (y_max - y_min) * i / tick_count
        y = y_pos(y_value)
        grid_lines.append(
            f"<line x1='{left}' y1='{y:.2f}' x2='{left + plot_w}' y2='{y:.2f}' "
            "stroke='#e5e7eb' stroke-width='1'/>"
        )
        grid_lines.append(
            f"<text x='{left - 10}' y='{y + 4:.2f}' text-anchor='end' font-size='12' fill='#374151'>{y_value:.3f}</text>"
        )

    x_ticks = []
    for iteration, _ in kernel_points:
        x = x_pos(iteration)
        x_ticks.append(
            f"<line x1='{x:.2f}' y1='{top}' x2='{x:.2f}' y2='{top + plot_h}' stroke='#f3f4f6' stroke-width='1'/>"
        )
        x_ticks.append(
            f"<text x='{x:.2f}' y='{top + plot_h + 24}' text-anchor='middle' font-size='12' fill='#374151'>{iteration}</text>"
        )

    kernel_polyline = " ".join(f"{x_pos(iteration):.2f},{y_pos(value):.2f}" for iteration, value in kernel_points)
    cusparse_y = y_pos(cusparse_ms)

    circles = []
    for iteration, value in kernel_points:
        x = x_pos(iteration)
        y = y_pos(value)
        circles.append(f"<circle cx='{x:.2f}' cy='{y:.2f}' r='4' fill='#2563eb'/>")
        circles.append(
            f"<text x='{x:.2f}' y='{y - 10:.2f}' text-anchor='middle' font-size='11' fill='#1d4ed8'>{value:.3f}</text>"
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
  <rect x="0" y="0" width="{width}" height="{height}" fill="white"/>
  <text x="{width/2:.0f}" y="24" text-anchor="middle" font-size="20" fill="#111827">ME_TCF Agent Feedback Performance Curve</text>
  <text x="{width/2:.0f}" y="46" text-anchor="middle" font-size="13" fill="#4b5563">Kernel time is a polyline; cuSPARSE time is a horizontal line.</text>
  {''.join(grid_lines)}
  {''.join(x_ticks)}
  <line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#111827" stroke-width="1.5"/>
  <line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#111827" stroke-width="1.5"/>
  <line x1="{left}" y1="{cusparse_y:.2f}" x2="{left + plot_w}" y2="{cusparse_y:.2f}" stroke="#dc2626" stroke-width="2" stroke-dasharray="8 6"/>
  <polyline fill="none" stroke="#2563eb" stroke-width="3" points="{kernel_polyline}"/>
  {''.join(circles)}
  <text x="{left + plot_w - 6}" y="{cusparse_y - 8:.2f}" text-anchor="end" font-size="12" fill="#dc2626">cuSPARSE avg {cusparse_ms:.3f} ms</text>
  <text x="{left + plot_w / 2:.2f}" y="{height - 24}" text-anchor="middle" font-size="14" fill="#111827">Iteration</text>
  <text x="24" y="{top + plot_h / 2:.2f}" text-anchor="middle" font-size="14" fill="#111827" transform="rotate(-90 24 {top + plot_h / 2:.2f})">Time (ms)</text>
  <rect x="{left}" y="{height - 54}" width="14" height="4" fill="#2563eb"/>
  <text x="{left + 22}" y="{height - 47}" font-size="12" fill="#111827">Generated kernel</text>
  <line x1="{left + 160}" y1="{height - 50}" x2="{left + 194}" y2="{height - 50}" stroke="#dc2626" stroke-width="2" stroke-dasharray="8 6"/>
  <text x="{left + 202}" y="{height - 47}" font-size="12" fill="#111827">cuSPARSE average</text>
</svg>
"""
    plot_path.write_text(svg)
    return plot_path


def main() -> None:
    args = parse_args()
    set_logging_level_for_all("DEBUG" if args.debug else "INFO")
    run_root = Path(args.loop_log_dir)
    if not run_root.is_absolute():
        run_root = REPO_ROOT / run_root
    run_dir = run_root / (
        datetime.now().strftime("%Y%m%d_%H%M%S")
        + f"_me_tcf_{args.strategy_provider}_feedback"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    strategy_defaults = StrategyConfig()
    base_strategy_config = StrategyConfig(
        provider=args.strategy_provider,
        log_dir=str(run_dir / "agent_runs"),
        api_key=args.strategy_api_key or strategy_defaults.api_key,
        model=args.strategy_model or strategy_defaults.model,
        api_base_url=strategy_defaults.api_base_url,
        anthropic_base_url=strategy_defaults.anthropic_base_url,
        timeout_sec=args.strategy_timeout_sec,
        temperature=strategy_defaults.temperature,
        max_tokens=strategy_defaults.max_tokens,
        anthropic_version=strategy_defaults.anthropic_version,
    )
    _write_json(run_dir / "run_config.json", vars(args))

    gmem_inouts = _build_gmem_inouts()
    history: List[Dict[str, Any]] = []
    previous_entry: Optional[Dict[str, Any]] = None
    latest_validation: Dict[str, Any] = {}
    latest_performance: Dict[str, Any] = {}
    latest_profile: Dict[str, Any] = {}
    current_kernel_code = ""
    current_kernel_path = ""

    if args.seed_heuristic_baseline:
        baseline_dir = run_dir / "baseline_heuristic"
        baseline_dir.mkdir(parents=True, exist_ok=True)
        baseline_config = StrategyConfig(**asdict(base_strategy_config))
        baseline_config.provider = "heuristic"
        baseline_config.log_dir = str(baseline_dir / "agent")
        baseline_decision, baseline_agent_log_dir = run_strategy_agent(
            format_name="ME_TCF",
            gmem_inouts=gmem_inouts,
            config=baseline_config,
        )
        baseline_inc = _generate_inc_for_strategy(decision=baseline_decision, iter_dir=baseline_dir)
        baseline_inc_path = baseline_dir / "me_tcf_kernel3.inc"
        shutil.copy2(baseline_inc, baseline_inc_path)
        current_kernel_path = str(baseline_inc_path)

        baseline_validation = _run_validation(baseline_dir, args)
        baseline_performance = _run_performance(baseline_dir, args)
        baseline_profile = _run_ncu_profile(baseline_dir, args)
        baseline_summary = {
            "iteration": 0,
            "label": "baseline_heuristic",
            "agent_log_dir": str(baseline_agent_log_dir),
            "strategy": _strategy_brief(baseline_decision),
            "validation": baseline_validation,
            "performance": baseline_performance,
            "profile_summary": baseline_profile,
        }
        _write_json(baseline_dir / "iteration_summary.json", baseline_summary)
        history.append(baseline_summary)
        previous_entry = baseline_summary
        latest_validation = baseline_validation
        latest_performance = baseline_performance
        latest_profile = baseline_profile

    for iteration in range(1, args.iterations + 1):
        iter_dir = run_dir / f"iter_{iteration:02d}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        best_entry = _select_best_iteration(history)
        feedback_context = StrategyFeedbackContext(
            iteration=iteration,
            max_iterations=args.iterations,
            history=history,
            best_iteration=best_entry,
            latest_validation=latest_validation,
            latest_performance=latest_performance,
            latest_profile=latest_profile,
            current_kernel_path=current_kernel_path,
            current_kernel_code=current_kernel_code,
            include_full_kernel_code=args.include_full_inc_code,
            additional_notes=_build_feedback_notes(
                previous_entry=previous_entry,
                best_entry=best_entry,
            ),
        )

        strategy_config = StrategyConfig(**asdict(base_strategy_config))
        strategy_config.log_dir = str(iter_dir / "agent")
        decision, agent_log_dir = _request_strategy_with_retry(
            format_name="ME_TCF",
            gmem_inouts=gmem_inouts,
            strategy_config=strategy_config,
            feedback_context=feedback_context,
            iter_dir=iter_dir,
        )
        (iter_dir / "strategy.json").write_text(
            json.dumps(decision.to_dict(), indent=2, ensure_ascii=False)
        )

        generated_inc = _generate_inc_for_strategy(decision=decision, iter_dir=iter_dir)
        iteration_inc_path = iter_dir / "me_tcf_kernel3.inc"
        shutil.copy2(generated_inc, iteration_inc_path)
        current_kernel_path = str(iteration_inc_path)

        validation = _run_validation(iter_dir, args)
        if validation.get("build_succeeded"):
            performance = _run_performance(iter_dir, args)
            profile_summary = _run_ncu_profile(iter_dir, args)
        else:
            performance = {
                "mykernel_ms": None,
                "cusparse_ms": None,
                "ratio_vs_cusparse": None,
                "speedup_vs_generated": None,
                "returncode": None,
                "raw_log": None,
            }
            profile_summary = {"available": False}

        iteration_summary = {
            "iteration": iteration,
            "agent_log_dir": str(agent_log_dir),
            "strategy": _strategy_brief(decision),
            "validation": validation,
            "performance": performance,
            "profile_summary": profile_summary,
        }
        _write_json(iter_dir / "iteration_summary.json", iteration_summary)

        history.append(iteration_summary)
        previous_entry = iteration_summary
        latest_validation = validation
        latest_performance = performance
        latest_profile = profile_summary

        logger.info(
            "Iteration %d/%d: passed=%s mykernel_ms=%s ratio_vs_cusparse=%s",
            iteration,
            args.iterations,
            validation.get("passed"),
            performance.get("mykernel_ms"),
            performance.get("ratio_vs_cusparse"),
        )

    best_entry = _select_best_iteration(history)
    _write_json(run_dir / "summary.json", {"history": history, "best_iteration": best_entry})
    curve_path = _write_curve_csv(run_dir, history)
    plot_path = _write_curve_plot(run_dir, history)
    logger.info(
        "Feedback loop finished. Summary=%s curve=%s plot=%s",
        run_dir / "summary.json",
        curve_path,
        plot_path,
    )


if __name__ == "__main__":
    main()
