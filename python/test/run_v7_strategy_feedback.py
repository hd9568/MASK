#!/usr/bin/env python3
"""Tune each executable v7 run-owned format with a strategy/NCU feedback loop.

Iteration 0 is always the deterministic heuristic baseline.  Iterations 1..K
are structured Strategy Agent decisions. Agent requests and dataset pipeline
waits may overlap across workers. The lowering/compiler critical section is
serialized because it contains process-global mutable state; it can still run
while another dataset is in a GPU measurement. GPU correctness, timing, and
NCU calls are protected by one process-wide GPU lock, while each dataset
remains sequential across its own iterations.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import shutil
from statistics import median
import threading
import time
from typing import Any

from generate_custom_format_operators import generate_one
from run_custom_format_e2e import compile_one, run_one
from sparsene.formats.custom.loader import load_run_format
from sparsene.op_gen.backend_registry import select_backend
from sparsene.op_gen.strategy_agent import (
    PipelineStrategy,
    RuntimeScheduleStrategy,
    StrategyConfig,
    StrategyDecision,
    StrategyFeedbackContext,
    TensorPlacementStrategy,
    normalize_strategy_for_backend,
    run_strategy_agent,
    strategy_search_space,
)
from sparsene.profiling.ncu import collect_ncu_profile, parse_ncu_csv


SCHEMA = "sparsene.v7_strategy_feedback.v1"
PERFORMANCE_CONTRACT = "sparsene.kernel_only_post_conversion.v1"
# Avoid CUDA's `bin/ncu` wrapper, which selects the newest installed NCU
# release. On the current A800/570.86.15 environment that wrapper now selects
# 2026.3, which cannot attach to the driver; the toolkit-pinned 2025.1 binary
# is compatible and has been verified against the generated-kernel runner.
DEFAULT_NCU = Path("/home/huoda/local/cuda/nsight-compute-2025.1.0/ncu")
REPO_ROOT = Path(__file__).resolve().parents[2]
IMPLEMENTATION_PATHS = (
    "python/test/run_v7_strategy_feedback.py",
    "python/test/generate_custom_format_operators.py",
    "python/test/run_custom_format_e2e.py",
    "python/sparsene/op_gen/strategy_agent.py",
    "python/sparsene/op_gen/backend_registry.py",
    "python/sparsene/op_gen/nvir/generate.py",
    "python/sparsene/op_gen/nvir/compiler_driver.py",
    "python/sparsene/op_gen/nvir/codegen.py",
    "python/sparsene/profiling/ncu.py",
    "examples/src_fp32/acc/testbed/CMakeLists.txt",
    "examples/src_fp32/acc/testbed/custom_format_e2e.cu",
    ".agent/skills/sparsene-framework/SKILL.md",
    ".agent/skills/strategy-optimization/SKILL.md",
    ".agent/skills/feedback-loop/SKILL.md",
)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable_fingerprint(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _implementation_hashes() -> dict[str, str]:
    return {
        relative: _sha256(REPO_ROOT / relative)
        for relative in IMPLEMENTATION_PATHS
    }


def _compact_features(features: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    candidate_key = f"{plan['logical_blk_m']}x{plan['logical_blk_k']}"
    return {
        "schema_version": features.get("schema_version"),
        "matrix_path": features.get("path"),
        "shape": features.get("shape"),
        "density": features.get("density"),
        "row_nnz_distribution": features.get("row_nnz_distribution"),
        "col_nnz_distribution": features.get("col_nnz_distribution"),
        "locality": features.get("locality"),
        "reorder_proxy": features.get("reorder_proxy"),
        "selected_hierarchical_candidate": features.get("hierarchical_candidates", {}).get(candidate_key),
    }


def _operator_stats(operator_path: Path) -> dict[str, Any]:
    source = operator_path.read_text(encoding="utf-8")
    return {
        "bytes": operator_path.stat().st_size,
        "sha256": _sha256(operator_path),
        "cp_async_sites": source.count("SM80_CP_ASYNC") + source.count("__pipeline_memcpy_async"),
        "pipeline_commit_sites": source.count("__pipeline_commit"),
        "pipeline_wait_sites": source.count("__pipeline_wait_prior"),
        "warp_sync_sites": source.count("SPARSENE_GENERATED_BLOCK_SYNC()") - 1,
        "atomic_add_sites": source.count("atomicAdd("),
        "tf32_mma_sites": source.count("mma_m16n8k8_fp32_tf32_tf32_fp32"),
        "has_launch_bounds": "__launch_bounds__" in source,
    }


def _strategy_signature(decision: StrategyDecision) -> str:
    payload = decision.to_dict()
    signature = {
        "tensor_placement": payload["tensor_placement"],
        "pipeline": payload["pipeline"],
        "compiler": payload["compiler"],
        "runtime_schedule": payload["runtime_schedule"],
    }
    return hashlib.sha256(
        json.dumps(signature, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _baseline_decision() -> StrategyDecision:
    return StrategyDecision(
        provider="heuristic",
        rationale="Iteration 0 conservative SM80 baseline for the fixed v7 format.",
        tensor_placement=TensorPlacementStrategy(),
        pipeline=PipelineStrategy(
            enable_async_pipeline=True,
            sync_threads_on_smem_input=True,
            max_wait_prior=None,
            stage_shifts=[],
            rationale="Default dependency-derived software pipeline.",
        ),
        runtime_schedule=RuntimeScheduleStrategy(
            compile_max_tiles_per_segment=32,
            runtime_max_tiles_per_segment=0,
            execution_policy="auto",
            task_group_tiles=64,
        ),
        agent_metadata={"source": "heuristic_iteration_0"},
    )


def _normalized_profile(entry: dict[str, Any]) -> dict[str, Any]:
    """Reparse retained CSV so old summaries also receive current unit fixes."""

    profile = dict(entry.get("ncu_profile", {}) or {})
    csv_path = profile.get("csv_path")
    if csv_path and Path(csv_path).is_file():
        reparsed = parse_ncu_csv(Path(csv_path).read_text(encoding="utf-8"))
        reparsed.update(
            {
                key: value
                for key, value in profile.items()
                if key in {"returncode", "raw_log", "csv_path", "command_path", "reason"}
            }
        )
        return reparsed
    # Compatibility for an old mixed-unit summary without its CSV.
    if (
        profile.get("memory_throughput_pct", 0) > 100
        and profile.get("units", {}).get("memory_throughput_pct") in {"byte/s", "bytes/s"}
    ):
        profile["memory_bandwidth_bytes_per_s"] = profile.pop("memory_throughput_pct")
        profile.setdefault("memory_throughput_pct", profile.get("dram_throughput_pct"))
    return profile


def _effective_execution(result: dict[str, Any]) -> dict[str, Any]:
    packed = result.get("packed", {})
    return {
        "execution_policy": packed.get("execution_policy"),
        "segmentation_policy": packed.get("segmentation_policy"),
        "max_tiles_per_execution_segment": packed.get("max_tiles_per_execution_segment"),
        "execution_segments": packed.get("execution_segments"),
        "task_groups": packed.get("task_groups"),
        "tasks": packed.get("tasks"),
        "task_group_tiles": packed.get("task_group_tiles"),
        "row_tiles_mean": packed.get("row_tiles_mean"),
        "row_tiles_p95": packed.get("row_tiles_p95"),
        "row_tiles_max": packed.get("row_tiles_max"),
        "distribution_heavy_tail": packed.get("distribution_heavy_tail"),
    }


def _compact_history(
    entry: dict[str, Any],
    *,
    baseline_ms: float | None = None,
    best_ms: float | None = None,
) -> dict[str, Any]:
    result = entry.get("e2e_result", {})
    timing = result.get("timing", {})
    correctness = result.get("correctness", {})
    generated = timing.get("generated_ms")
    cusparse = timing.get("cusparse_ms")
    performance = {
        "mykernel_ms": generated,
        "cusparse_ms": cusparse,
        "ratio_vs_cusparse": generated / cusparse if generated and cusparse else None,
        "speedup_vs_cusparse": cusparse / generated if generated and cusparse else None,
        "speedup_vs_baseline": baseline_ms / generated if generated and baseline_ms else None,
        "delta_pct_vs_baseline": (
            (generated / baseline_ms - 1.0) * 100.0
            if generated and baseline_ms else None
        ),
        "delta_pct_vs_best": (
            (generated / best_ms - 1.0) * 100.0
            if generated and best_ms else None
        ),
        "measurement_runs": timing.get("measurement_runs", 1),
        "generated_ms_samples": timing.get("generated_ms_samples", [generated] if generated else []),
        "relative_spread_pct": timing.get("relative_spread_pct"),
    }
    executable_sha256 = entry.get("executable_sha256")
    executable_path = entry.get("executable")
    if not executable_sha256 and executable_path and Path(executable_path).is_file():
        executable_sha256 = _sha256(Path(executable_path))
    return {
        "iteration": entry["iteration"],
        "strategy": entry.get("strategy"),
        "strategy_signature": entry.get("strategy_signature"),
        "validation": {
            "passed": correctness.get("passed") is True,
            "mismatches": correctness.get("mismatches"),
            "nonfinite": correctness.get("nonfinite"),
            "relative_l2_error": correctness.get("relative_l2_error"),
        },
        "performance": performance,
        "requested_execution": entry.get("strategy", {}).get("runtime_schedule", {}),
        "effective_execution": _effective_execution(result),
        "profile_summary": _normalized_profile(entry),
        "code_stats": entry.get("operator_stats", {}),
        "artifact_identity": {
            "operator_sha256": entry.get("operator_stats", {}).get("sha256"),
            "executable_sha256": executable_sha256,
        },
        "status": entry.get("status"),
        "error": entry.get("error", ""),
    }


def _value_key(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _strategy_coverage(entries: list[dict[str, Any]]) -> dict[str, Any]:
    space = strategy_search_space()
    packing_mode = None
    for entry in entries:
        packing_mode = entry.get("e2e_result", {}).get("format", {}).get("packing_mode")
        if packing_mode:
            break
    inactive = set()
    if packing_mode != "coo":
        inactive.update(
            {
                "tensor_placement.coo_idx_output_mem",
                "tensor_placement.coo_val_output_mem",
                "tensor_placement.coo_load_mode",
                "tensor_placement.coo_restore_swizzle",
            }
        )
    coverage: dict[str, Any] = {
        "packing_mode": packing_mode,
        "fields": {},
    }
    for group in ("tensor_placement", "pipeline", "compiler", "runtime_schedule"):
        for field_name, choices in space[group].items():
            dotted = f"{group}.{field_name}"
            tested_by_key: dict[str, Any] = {}
            for entry in entries:
                strategy = entry.get("strategy", {})
                if field_name in strategy.get(group, {}):
                    value = strategy[group][field_name]
                    tested_by_key[_value_key(value)] = value
            untested = [choice for choice in choices if _value_key(choice) not in tested_by_key]
            coverage["fields"][dotted] = {
                "status": (
                    "inactive_for_backend" if dotted in inactive
                    else "fixed" if len(choices) == 1
                    else "searchable"
                ),
                "tested": list(tested_by_key.values()),
                "untested": untested,
                "covered_choices": len(tested_by_key),
                "total_choices": len(choices),
            }
    searchable = [
        item for item in coverage["fields"].values() if item["status"] == "searchable"
    ]
    coverage["searchable_choice_coverage"] = {
        "tested": sum(item["covered_choices"] for item in searchable),
        "total": sum(item["total_choices"] for item in searchable),
    }
    return coverage


def _experiment_guidance(
    *, iteration: int, history: list[dict[str, Any]], coverage: dict[str, Any]
) -> dict[str, Any]:
    best = min(
        (item for item in history if item.get("validation", {}).get("passed")),
        key=lambda item: item.get("performance", {}).get("mykernel_ms", float("inf")),
        default=None,
    )
    best_profile = (best or {}).get("profile_summary", {})
    best_effective = (best or {}).get("effective_execution", {})
    priorities = [
        "runtime_schedule.task_group_tiles",
        "runtime_schedule.runtime_max_tiles_per_segment",
        "compiler.max_registers_per_thread",
        "pipeline.max_wait_prior",
        "pipeline.stage_shifts",
        "compiler.launch_bounds_min_blocks_per_sm",
        "tensor_placement.materialize_b_array_ref_to_smem",
        "pipeline.sync_threads_on_smem_input",
    ]
    suggested = []
    for dotted in priorities:
        item = coverage.get("fields", {}).get(dotted, {})
        if item.get("status") == "searchable" and item.get("untested"):
            suggested.append({"field": dotted, "untested_values": item["untested"]})
        if len(suggested) >= 4:
            break
    notes = [
        "Use one primary hypothesis per iteration and retain the best-correct values for unrelated fields.",
        "Do not treat occupancy improvement alone as success; the warmed median kernel time is authoritative.",
    ]
    if best_effective.get("execution_policy") == "strict_lb":
        notes.append(
            "The current best requested auto but already resolved to strict_lb; explicit strict_lb with the same task_group_tiles is effectively redundant."
        )
    if (
        (best_profile.get("dram_throughput_pct") or 0) >= 60
        and (best_profile.get("compute_throughput_pct") or 100) < 40
    ):
        notes.append(
            "The best kernel is memory-dominant. Preserve tile_b=64/B reuse and aligned cp.async unless testing that exact data-movement hypothesis."
        )
    return {
        "iteration": iteration,
        "suggested_untested_axes": suggested,
        "notes": notes,
    }


def _best_correct(entries: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [
        entry for entry in entries
        if entry.get("status") == "validated"
        and entry.get("e2e_result", {}).get("correctness", {}).get("passed") is True
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda item: item["e2e_result"]["timing"]["generated_ms"])


def _feedback_context(
    *,
    iteration: int,
    agent_iterations: int,
    entries: list[dict[str, Any]],
    measurement_runs: int,
    warmup: int,
    repeat: int,
) -> StrategyFeedbackContext:
    baseline_ms = (
        entries[0].get("e2e_result", {}).get("timing", {}).get("generated_ms")
        if entries else None
    )
    best = _best_correct(entries)
    best_ms = (
        best.get("e2e_result", {}).get("timing", {}).get("generated_ms")
        if best else None
    )
    history = [
        _compact_history(item, baseline_ms=baseline_ms, best_ms=best_ms)
        for item in entries
    ]
    latest = history[-1] if history else {}
    best_compact = (
        _compact_history(best, baseline_ms=baseline_ms, best_ms=best_ms)
        if best else None
    )
    coverage = _strategy_coverage(entries)
    scoreboard = [
        {
            "iteration": item["iteration"],
            "status": item.get("status"),
            "mykernel_ms": item.get("performance", {}).get("mykernel_ms"),
            "speedup_vs_baseline": item.get("performance", {}).get("speedup_vs_baseline"),
            "delta_pct_vs_best": item.get("performance", {}).get("delta_pct_vs_best"),
            "measurement_runs": item.get("performance", {}).get("measurement_runs"),
            "relative_spread_pct": item.get("performance", {}).get("relative_spread_pct"),
            "effective_execution": item.get("effective_execution"),
            "operator_sha256": item.get("artifact_identity", {}).get("operator_sha256"),
            "executable_sha256": item.get("artifact_identity", {}).get("executable_sha256"),
        }
        for item in history
    ]
    return StrategyFeedbackContext(
        iteration=iteration,
        max_iterations=agent_iterations,
        history=history,
        best_iteration=best_compact,
        latest_validation=latest.get("validation", {}),
        correctness_feedback=latest.get("validation", {}),
        latest_performance=latest.get("performance", {}),
        latest_profile=latest.get("profile_summary", {}),
        scoreboard=scoreboard,
        strategy_coverage=coverage,
        experiment_guidance=_experiment_guidance(
            iteration=iteration, history=history, coverage=coverage
        ),
        measurement_policy={
            "measurement_runs": measurement_runs,
            "ranking_statistic": "median_generated_ms",
            "per_measurement_warmup": warmup,
            "per_measurement_repeat": repeat,
            "noise_threshold_pct": 1.5,
            "rule": "A sub-1.5% change is inconclusive without repeated samples and should not justify a broad strategy change.",
        },
        current_kernel_path=best.get("operator_path") if best else None,
        include_full_kernel_code=False,
        additional_notes=[
            "Iteration 0 is the mandatory heuristic performance floor.",
            "Do not repeat an existing normalized strategy signature unless profile evidence predicts stability is preferable.",
            "Only a full-output-correct candidate can become best.",
            "Requested execution knobs and effective packed execution are different contracts; reason from the effective fields.",
        ],
    )


def _strategy_config(args: argparse.Namespace, iteration_dir: Path) -> StrategyConfig:
    return StrategyConfig(
        provider=args.strategy_provider,
        model_name=args.strategy_model,
        model=args.strategy_model,
        log_dir=str(iteration_dir / "agent"),
        api_key=None,
        api_base_url=None,
        anthropic_base_url=None,
        timeout_sec=args.strategy_timeout_sec,
        temperature=args.strategy_temperature,
        max_tokens=args.strategy_max_tokens,
        model_config_path=str(args.model_config.resolve()),
    )


def _v7_strategy_search_space() -> dict[str, Any]:
    """Expose only B-load choices implemented by the v7 semantic fusion.

    The current v7 B-load fusion always gathers B into shared memory using its
    fixed SM80 cp.async implementation.  Advertising direct-GMEM or synchronous
    B variants lets the Agent request a strategy that the fused lowering does
    not implement (and can produce an incorrect output tensor contract).
    Keep this restriction local to the v7 fixed-format route.
    """

    space = strategy_search_space()
    fixed_values = {
        "tensor_placement": {
            "materialize_b_array_ref_to_smem": [True],
            "enable_b_array_ref_cp_async": [True],
            "b_array_ref_mode": ["cp_async_1x4"],
        },
        "pipeline": {
            "enable_async_pipeline": [True],
        },
    }
    rationale = (
        "Fixed by the v7 semantic B-load fusion: it writes the gathered B tile "
        "to shared memory with the SM80 cp.async implementation."
    )
    for group, fields in fixed_values.items():
        for field_name, choices in fields.items():
            space[group][field_name] = choices
            metadata = space["field_catalog"][group][field_name]
            metadata["choices"] = choices
            metadata["applicability"] = rationale
    raw_size = 1
    for group in ("tensor_placement", "pipeline", "compiler", "runtime_schedule"):
        for choices in space[group].values():
            raw_size *= len(choices)
    space["raw_cartesian_size"] = raw_size
    space["note"] += " v7 fixes B gather materialization/mode to the implemented fused SM80 path."
    return space


def _normalize_v7_strategy(
    decision: StrategyDecision, *, packing_mode: str
) -> StrategyDecision:
    """Apply backend normalization plus the v7 B-load fusion contract."""

    payload = decision.to_dict()
    tensor = payload.setdefault("tensor_placement", {})
    pipeline = payload.setdefault("pipeline", {})
    original = {
        "tensor_placement": {
            name: tensor.get(name)
            for name in (
                "materialize_b_array_ref_to_smem",
                "enable_b_array_ref_cp_async",
                "b_array_ref_mode",
            )
        },
        "pipeline": {"enable_async_pipeline": pipeline.get("enable_async_pipeline")},
    }
    tensor.update(
        {
            "materialize_b_array_ref_to_smem": True,
            "enable_b_array_ref_cp_async": True,
            "b_array_ref_mode": "cp_async_1x4",
        }
    )
    pipeline["enable_async_pipeline"] = True
    normalized = normalize_strategy_for_backend(
        StrategyDecision.from_dict(payload), packing_mode=packing_mode
    )
    metadata = dict(normalized.agent_metadata)
    backend_metadata = dict(metadata.get("backend_normalization", {}))
    backend_metadata["v7_b_load_contract"] = {
        "required": {
            "materialize_b_array_ref_to_smem": True,
            "enable_b_array_ref_cp_async": True,
            "b_array_ref_mode": "cp_async_1x4",
            "enable_async_pipeline": True,
        },
        "agent_values_before_normalization": original,
    }
    metadata["backend_normalization"] = backend_metadata
    normalized.agent_metadata = metadata
    return normalized


def _is_transient_strategy_error(exc: Exception) -> bool:
    """Recognize transport interruptions without retrying schema/model errors."""

    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    message = str(exc).lower()
    transient_markers = (
        "unexpected_eof_while_reading",
        "eof occurred in violation of protocol",
        "could not reach endpoint",
        "streaminginterruptederror",
        "stream interrupted",
        "sse ended before a terminal",
        "remote end closed connection",
        "remote connection closed",
        "connection reset by peer",
        "temporarily unavailable",
        "failed with status=429",
        "failed with status=500",
        "failed with status=502",
        "failed with status=503",
        "failed with status=504",
    )
    return any(marker in message for marker in transient_markers)


def _call_strategy_with_retry(
    request_fn,
    *,
    audit_path: Path,
    max_attempts: int = 3,
    backoff_base_sec: float = 2.0,
):
    """Retry transient OpenLux transport errors; every invocation is audited.

    ``invoke_json_agent`` creates a separate timestamped audit directory for
    every call, including interrupted streams. This small outer retry only
    repeats the strategy request, never codegen, compilation, or GPU work.
    """

    attempts: list[dict[str, Any]] = []
    for attempt in range(1, max_attempts + 1):
        try:
            result = request_fn()
        except Exception as exc:
            retryable = _is_transient_strategy_error(exc)
            delay = min(backoff_base_sec * (2 ** (attempt - 1)), 8.0)
            attempts.append(
                {
                    "attempt": attempt,
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "retryable": retryable,
                    "backoff_sec": delay if retryable and attempt < max_attempts else 0,
                }
            )
            _write_json(audit_path, {"status": "retrying" if retryable and attempt < max_attempts else "failed",
                                     "max_attempts": max_attempts, "attempts": attempts})
            if not retryable or attempt >= max_attempts:
                raise
            time.sleep(delay)
        else:
            if attempts:
                attempts.append({"attempt": attempt, "status": "succeeded"})
                _write_json(audit_path, {"status": "succeeded_after_retry", "max_attempts": max_attempts,
                                         "attempts": attempts})
            return result
    raise AssertionError("retry loop exited without a result or exception")


def _design_context(
    *,
    dataset: str,
    spec: dict[str, Any],
    artifact_path: Path,
    artifact_sha: str,
    plan: dict[str, Any],
    features: dict[str, Any],
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "format_artifact": {
            "path": str(artifact_path),
            "sha256": artifact_sha,
            "format_name": spec.get("format_name"),
            "atomic_type": spec.get("atomic", {}).get("type"),
            "params": spec.get("params"),
            "levels": spec.get("levels"),
        },
        "backend_plan_before_strategy": plan,
        "matrix_features": _compact_features(features, plan),
        "objective": PERFORMANCE_CONTRACT,
        "fixed_problem": {"N": 512, "input_output": "fp32", "mma": "tf32", "target": "sm80"},
    }


def _profile(
    *,
    args: argparse.Namespace,
    dataset: str,
    features: dict[str, Any],
    executable: Path,
    decision: StrategyDecision,
    iteration_dir: Path,
) -> dict[str, Any]:
    if args.skip_ncu:
        return {"available": False, "reason": "disabled_by_cli"}
    schedule = decision.runtime_schedule
    ncu_result = iteration_dir / "ncu" / "probe_result.json"
    executable_args = [
        "--filename", str(features["path"]),
        "--dataset", dataset,
        "--result-json", str(ncu_result),
        "--n", str(args.n),
        "--warmup", "0",
        "--repeat", "1",
        "--max-tiles-per-segment", str(schedule.runtime_max_tiles_per_segment),
        "--execution-policy", schedule.execution_policy,
        "--task-group-tiles", str(schedule.task_group_tiles),
    ]
    return collect_ncu_profile(
        ncu_bin=args.ncu_bin,
        executable=executable,
        executable_args=executable_args,
        output_dir=iteration_dir / "ncu",
        timeout_sec=args.ncu_timeout_sec,
    )


def _aggregate_measurements(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("measurement sample list is empty")
    aggregate = deepcopy(samples[0])
    generated_samples = [float(item["timing"]["generated_ms"]) for item in samples]
    cusparse_samples = [float(item["timing"]["cusparse_ms"]) for item in samples]
    generated_median = float(median(generated_samples))
    cusparse_median = float(median(cusparse_samples))
    timing = aggregate["timing"]
    timing.update(
        {
            "measurement_runs": len(samples),
            "ranking_statistic": "median",
            "generated_ms": generated_median,
            "cusparse_ms": cusparse_median,
            "speedup_vs_cusparse": cusparse_median / generated_median,
            "generated_ms_samples": generated_samples,
            "cusparse_ms_samples": cusparse_samples,
            "generated_ms_min": min(generated_samples),
            "generated_ms_max": max(generated_samples),
            "relative_spread_pct": (
                (max(generated_samples) - min(generated_samples)) / generated_median * 100.0
                if generated_median else None
            ),
        }
    )
    correctness_samples = [item.get("correctness", {}) for item in samples]
    aggregate["correctness"] = dict(correctness_samples[0])
    aggregate["correctness"].update(
        {
            "passed": all(item.get("passed") is True for item in correctness_samples),
            "measurement_runs": len(samples),
            "all_measurements_passed": all(
                item.get("passed") is True for item in correctness_samples
            ),
            "max_mismatches_across_measurements": max(
                int(item.get("mismatches", 0)) for item in correctness_samples
            ),
            "max_nonfinite_across_measurements": max(
                int(item.get("nonfinite", 0)) for item in correctness_samples
            ),
        }
    )
    aggregate["measurement_sample_paths"] = [item.get("result_path") for item in samples]
    aggregate["measurement_ids"] = [item.get("measurement_id") for item in samples]
    return aggregate


def _evaluate_iteration(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    tuning_dir: Path,
    dataset: str,
    features: dict[str, Any],
    decision: StrategyDecision,
    iteration: int,
    campaign_fingerprint: str,
    gpu_lock: threading.Lock | None = None,
    codegen_lock: threading.Lock | None = None,
) -> dict[str, Any]:
    iteration_dir = tuning_dir / f"iteration_{iteration:02d}"
    iteration_dir.mkdir(parents=True, exist_ok=True)
    _write_json(iteration_dir / "strategy_decision.json", decision.to_dict())
    entry: dict[str, Any] = {
        "schema_version": SCHEMA,
        "dataset": dataset,
        "iteration": iteration,
        "provider": decision.provider,
        "strategy": decision.to_dict(),
        "strategy_signature": _strategy_signature(decision),
        "iteration_dir": str(iteration_dir.resolve()),
        "status": "failed",
        "performance_contract": PERFORMANCE_CONTRACT,
        "campaign_fingerprint": campaign_fingerprint,
    }
    try:
        schedule = decision.runtime_schedule
        def generate_and_compile() -> tuple[dict[str, Any], Path]:
            manifest = generate_one(
                run_dir,
                dataset,
                tile_n=decision.tensor_placement.tile_b,
                max_tiles_per_execution_segment=schedule.compile_max_tiles_per_segment,
                candidate_id="agent",
                worktree_label=f"strategy_iter_{iteration:02d}",
                strategy_decision=decision,
                work_dir_override=iteration_dir,
            )
            return manifest, compile_one(run_dir, manifest, jobs=1)

        # The lowering/compiler stack has process-global mutable state.  Keep
        # this small critical section serialized inside a process while still
        # allowing Agent calls and GPU samples from other datasets to overlap.
        if codegen_lock is None:
            manifest, executable = generate_and_compile()
        else:
            with codegen_lock:
                manifest, executable = generate_and_compile()
        entry["generation_manifest"] = str((iteration_dir / "generation_manifest.json").resolve())
        entry["operator_path"] = manifest["operator_path"]
        entry["operator_stats"] = _operator_stats(Path(manifest["operator_path"]))
        entry["executable"] = str(executable.resolve())
        entry["executable_sha256"] = _sha256(executable)
        samples = []
        for measurement_index in range(args.measurement_runs):
            if gpu_lock is None:
                sample = run_one(
                    run_dir,
                    manifest,
                    executable,
                    n=args.n,
                    warmup=args.warmup,
                    repeat=args.repeat,
                    runtime_max_tiles_per_segment=schedule.runtime_max_tiles_per_segment,
                    execution_policy=schedule.execution_policy,
                    task_group_tiles=schedule.task_group_tiles,
                    result_label=f"measurement_{measurement_index:02d}",
                )
            else:
                with gpu_lock:
                    sample = run_one(
                        run_dir,
                        manifest,
                        executable,
                        n=args.n,
                        warmup=args.warmup,
                        repeat=args.repeat,
                        runtime_max_tiles_per_segment=schedule.runtime_max_tiles_per_segment,
                        execution_policy=schedule.execution_policy,
                        task_group_tiles=schedule.task_group_tiles,
                        result_label=f"measurement_{measurement_index:02d}",
                    )
            samples.append(sample)
        entry["measurement_samples"] = [
            {
                "measurement_id": item.get("measurement_id"),
                "result_path": item.get("result_path"),
                "timing": item.get("timing"),
                "correctness": item.get("correctness"),
            }
            for item in samples
        ]
        result = _aggregate_measurements(samples)
        entry["e2e_result"] = result
        _write_json(iteration_dir / "e2e_result.json", result)
        if result["correctness"]["passed"]:
            entry["status"] = "validated"
            if gpu_lock is None:
                entry["ncu_profile"] = _profile(
                    args=args,
                    dataset=dataset,
                    features=features,
                    executable=executable,
                    decision=decision,
                    iteration_dir=iteration_dir,
                )
            else:
                with gpu_lock:
                    entry["ncu_profile"] = _profile(
                        args=args,
                        dataset=dataset,
                        features=features,
                        executable=executable,
                        decision=decision,
                        iteration_dir=iteration_dir,
                    )
        else:
            entry["status"] = "incorrect"
            entry["ncu_profile"] = {"available": False, "reason": "correctness_gate_failed"}
    except Exception as exc:
        entry["error"] = f"{type(exc).__name__}: {exc}"
    _write_json(iteration_dir / "iteration_summary.json", entry)
    return entry


def _publish_dataset_summary(
    tuning_dir: Path,
    dataset: str,
    entries: list[dict[str, Any]],
) -> dict[str, Any]:
    best = _best_correct(entries)
    summary = {
        "schema_version": SCHEMA,
        "dataset": dataset,
        "iterations": entries,
        "counts": {
            "total": len(entries),
            "validated": sum(item.get("status") == "validated" for item in entries),
            "incorrect": sum(item.get("status") == "incorrect" for item in entries),
            "failed": sum(item.get("status") == "failed" for item in entries),
        },
        "best_iteration": best.get("iteration") if best else None,
        "best_generated_ms": best.get("e2e_result", {}).get("timing", {}).get("generated_ms") if best else None,
        "best_speedup_vs_cusparse": best.get("e2e_result", {}).get("timing", {}).get("speedup_vs_cusparse") if best else None,
        "baseline_generated_ms": entries[0].get("e2e_result", {}).get("timing", {}).get("generated_ms") if entries else None,
        "improvement_vs_baseline": None,
        "best_operator_path": best.get("operator_path") if best else None,
    }
    baseline_ms = summary["baseline_generated_ms"]
    best_ms = summary["best_generated_ms"]
    if baseline_ms and best_ms:
        summary["improvement_vs_baseline"] = baseline_ms / best_ms
    _write_json(tuning_dir / "summary.json", summary)
    if best:
        shutil.copy2(best["operator_path"], tuning_dir / "best_operator.inc")
        _write_json(tuning_dir / "best_iteration.json", best)
    return summary


def _publish_global(
    run_dir: Path,
    summaries: list[dict[str, Any]],
    settings: dict[str, Any],
    tuning_name: str,
) -> None:
    payload = {
        "schema_version": SCHEMA,
        "performance_contract": PERFORMANCE_CONTRACT,
        "settings": settings,
        "counts": {
            "datasets": len(summaries),
            "completed": sum(item.get("status") == "completed" for item in summaries),
            "completed_with_failures": sum(
                item.get("status") == "completed_with_failures" for item in summaries
            ),
            "failed": sum(item.get("status") == "failed" for item in summaries),
            "unsupported": sum(item.get("status") == "unsupported" for item in summaries),
        },
        "results": summaries,
    }
    prefix = "strategy_tuning" if tuning_name == "strategy_tuning" else tuning_name
    _write_json(run_dir / f"{prefix}_summary.json", payload)
    fields = [
        "dataset", "status", "best_iteration", "baseline_generated_ms",
        "best_generated_ms", "improvement_vs_baseline", "best_speedup_vs_cusparse",
        "best_operator_path", "error",
    ]
    with (run_dir / f"{prefix}_summary.csv").open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summaries)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument(
        "--tuning-name",
        default="strategy_tuning",
        help="dataset-local output directory and run-level summary prefix",
    )
    parser.add_argument("--first", type=int)
    parser.add_argument("--exclude-datasets", nargs="+", default=[])
    parser.add_argument("--agent-iterations", type=int, default=5)
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=32)
    parser.add_argument(
        "--measurement-runs",
        type=int,
        default=3,
        help="independent warmup/repeat measurements per compiled strategy; rank by median",
    )
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument(
        "--dataset-workers",
        type=int,
        default=2,
        help="overlap dataset/Agent pipelines; codegen is protected by a process lock and GPU evaluation remains locked",
    )
    parser.add_argument("--strategy-provider", default="openai_http")
    parser.add_argument("--strategy-model", default="deepseek-v4-pro")
    parser.add_argument("--model-config", type=Path, default=Path("llm_models.yaml"))
    parser.add_argument("--strategy-timeout-sec", type=int, default=300)
    parser.add_argument("--strategy-temperature", type=float, default=0.2)
    parser.add_argument(
        "--strategy-max-tokens",
        type=int,
        default=16384,
        help="maximum model completion budget; iterative rounds return compact patches",
    )
    parser.add_argument("--ncu-bin", type=Path, default=DEFAULT_NCU)
    parser.add_argument("--ncu-timeout-sec", type=int, default=900)
    parser.add_argument("--skip-ncu", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="validate artifacts/backends and write the plan without model, build, or GPU calls")
    return parser.parse_args()


def _run_dataset(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    dataset: str,
    position: int,
    total_datasets: int,
    campaign_fingerprint: str,
    gpu_lock: threading.Lock,
    codegen_lock: threading.Lock,
) -> dict[str, Any]:
    """Run one dataset pipeline; GPU calls are serialized by ``gpu_lock``."""

    print(f"[{position:02d}/{total_datasets:02d}] {dataset}", flush=True)
    tuning_dir = run_dir / "datasets" / dataset / args.tuning_name
    try:
        spec, _format, artifact_path, artifact_sha = load_run_format(run_dir, dataset)
        base_plan = select_backend(
            spec, target="sm80", tile_n=64, max_tiles_per_execution_segment=32
        )
    except Exception as exc:
        summary = {
            "dataset": dataset,
            "status": "unsupported",
            "error": str(exc),
            "best_iteration": None,
        }
        _write_json(tuning_dir / "summary.json", summary)
        return summary

    features = _read_json(run_dir / "datasets" / dataset / "matrix_features.json")
    context = _design_context(
        dataset=dataset,
        spec=spec,
        artifact_path=artifact_path,
        artifact_sha=artifact_sha,
        plan=base_plan.to_dict(),
        features=features,
    )
    _write_json(tuning_dir / "design_context.json", context)
    _write_json(tuning_dir / "search_space.json", _v7_strategy_search_space())
    if args.dry_run:
        summary = {
            "dataset": dataset,
            "status": "planned",
            "best_iteration": None,
            "backend": base_plan.backend,
        }
        _write_json(tuning_dir / "summary.json", summary)
        return summary

    entries: list[dict[str, Any]] = []
    for iteration in range(args.agent_iterations + 1):
        iteration_dir = tuning_dir / f"iteration_{iteration:02d}"
        summary_path = iteration_dir / "iteration_summary.json"
        resumed_decision: StrategyDecision | None = None
        if args.resume and summary_path.is_file():
            cached = _read_json(summary_path)
            if cached.get("campaign_fingerprint") == campaign_fingerprint:
                # Correct and numerically rejected observations are complete,
                # reproducible terminal outcomes.  A transient build/GPU/API
                # failure is not: retry it instead of letting --resume freeze
                # an incomplete campaign forever.  When the failed evaluation
                # already contains a normalized strategy, reuse that decision
                # so retrying does not spend another Agent call.
                if cached.get("status") in {"validated", "incorrect"}:
                    entries.append(cached)
                    continue
                if isinstance(cached.get("strategy"), dict):
                    resumed_decision = _normalize_v7_strategy(
                        StrategyDecision.from_dict(cached["strategy"]),
                        packing_mode=base_plan.packing_mode,
                    )
                _write_json(
                    iteration_dir / "resume_retry.json",
                    {
                        "reason": "cached_iteration_not_terminal",
                        "cached_status": cached.get("status"),
                        "reuse_normalized_strategy": resumed_decision is not None,
                        "campaign_fingerprint": campaign_fingerprint,
                    },
                )
            else:
                _write_json(
                    iteration_dir / "resume_rejected.json",
                    {
                        "reason": "campaign_fingerprint_mismatch",
                        "cached": cached.get("campaign_fingerprint"),
                        "current": campaign_fingerprint,
                    },
                )
        if resumed_decision is not None:
            decision = resumed_decision
        elif iteration == 0:
            decision = _normalize_v7_strategy(
                _baseline_decision(), packing_mode=base_plan.packing_mode
            )
        else:
            baseline_manifest_path = tuning_dir / "iteration_00" / "generation_manifest.json"
            if not baseline_manifest_path.is_file():
                failure = {
                    "schema_version": SCHEMA,
                    "dataset": dataset,
                    "iteration": iteration,
                    "provider": args.strategy_provider,
                    "status": "failed",
                    "error": "heuristic baseline did not produce a generation manifest",
                    "iteration_dir": str(iteration_dir.resolve()),
                    "campaign_fingerprint": campaign_fingerprint,
                }
                _write_json(summary_path, failure)
                entries.append(failure)
                continue
            baseline_manifest = _read_json(baseline_manifest_path)
            feedback = _feedback_context(
                iteration=iteration,
                agent_iterations=args.agent_iterations,
                entries=entries,
                measurement_runs=args.measurement_runs,
                warmup=args.warmup,
                repeat=args.repeat,
            )
            try:
                decision, agent_dir = _call_strategy_with_retry(
                    lambda: run_strategy_agent(
                        format_name=base_plan.codegen_format_name,
                        gmem_inouts=baseline_manifest.get("gmem_inouts", {}),
                        config=_strategy_config(args, iteration_dir),
                        feedback_context=feedback,
                        design_context=context,
                        search_space=_v7_strategy_search_space(),
                    ),
                    audit_path=iteration_dir / "strategy_request_retries.json",
                )
            except Exception as exc:
                failure = {
                    "schema_version": SCHEMA,
                    "dataset": dataset,
                    "iteration": iteration,
                    "provider": args.strategy_provider,
                    "status": "failed",
                    "error": f"strategy_agent: {type(exc).__name__}: {exc}",
                    "iteration_dir": str(iteration_dir.resolve()),
                    "campaign_fingerprint": campaign_fingerprint,
                }
                _write_json(summary_path, failure)
                entries.append(failure)
                _publish_dataset_summary(tuning_dir, dataset, entries)
                continue
            decision.agent_metadata = dict(decision.agent_metadata)
            decision.agent_metadata["agent_run_dir"] = str(agent_dir)
            decision = _normalize_v7_strategy(
                decision, packing_mode=base_plan.packing_mode
            )
            previous = {
                _strategy_signature(StrategyDecision.from_dict(item["strategy"]))
                for item in entries
                if isinstance(item.get("strategy"), dict)
            }
            decision.agent_metadata["duplicate_normalized_strategy"] = (
                _strategy_signature(decision) in previous
            )
        entry = _evaluate_iteration(
            args=args,
            run_dir=run_dir,
            tuning_dir=tuning_dir,
            dataset=dataset,
            features=features,
            decision=decision,
            iteration=iteration,
            campaign_fingerprint=campaign_fingerprint,
            gpu_lock=gpu_lock,
            codegen_lock=codegen_lock,
        )
        entries.append(entry)
        _publish_dataset_summary(tuning_dir, dataset, entries)

    summary = _publish_dataset_summary(tuning_dir, dataset, entries)
    if summary.get("best_iteration") is None:
        summary["status"] = "failed"
    elif summary.get("counts", {}).get("failed", 0):
        summary["status"] = "completed_with_failures"
    else:
        summary["status"] = "completed"
    _write_json(tuning_dir / "summary.json", summary)
    return summary


def main() -> int:
    args = parse_args()
    if args.jobs != 1:
        raise SystemExit("strategy feedback requires --jobs 1; each worker already uses one compile job")
    if args.dataset_workers < 1:
        raise SystemExit("--dataset-workers must be positive")
    if args.n != 512:
        raise SystemExit("this v7 campaign is fixed to --n 512")
    if args.warmup < 0 or args.repeat < 1:
        raise SystemExit("--warmup must be nonnegative and --repeat must be positive")
    if args.agent_iterations < 0:
        raise SystemExit("--agent-iterations must be nonnegative")
    if args.measurement_runs < 1:
        raise SystemExit("--measurement-runs must be positive")
    if not args.tuning_name or any(
        char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
        for char in args.tuning_name
    ):
        raise SystemExit("--tuning-name must contain only letters, digits, '_' or '-'")
    run_dir = args.run_dir.resolve()
    manifest = _read_json(run_dir / "run_manifest.json")
    datasets = args.datasets or list(manifest["datasets"])
    excluded = set(args.exclude_datasets)
    datasets = [dataset for dataset in datasets if dataset not in excluded]
    if args.first is not None:
        datasets = datasets[: args.first]
    if not datasets:
        raise SystemExit("no datasets remain after --datasets/--exclude-datasets filtering")
    implementation_hashes = _implementation_hashes()
    settings = {
        "n": args.n,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "measurement_runs": args.measurement_runs,
        "ranking_statistic": "median_generated_ms",
        "jobs": args.jobs,
        "dataset_workers": args.dataset_workers,
        "agent_iterations": args.agent_iterations,
        "total_iterations_per_executable_dataset": args.agent_iterations + 1,
        "strategy_provider": args.strategy_provider,
        "strategy_model": args.strategy_model,
        "strategy_max_tokens": args.strategy_max_tokens,
        "strategy_response_mode": "delta_patch_for_feedback",
        "tuning_name": args.tuning_name,
        "model_config_path": str(args.model_config.resolve()),
        "model_config_sha256": (
            _sha256(args.model_config.resolve())
            if args.model_config.resolve().is_file()
            else None
        ),
        "ncu_bin": str(args.ncu_bin),
        "ncu_enabled": not args.skip_ncu,
        "dry_run": args.dry_run,
        "serial_gpu_execution": True,
        "gpu_scheduler": "dataset_workers_with_process_gpu_lock",
        "search_space": _v7_strategy_search_space(),
        "implementation_sha256": _stable_fingerprint(implementation_hashes),
        "implementation_files": implementation_hashes,
    }
    fingerprint_payload = dict(settings)
    fingerprint_payload.pop("dry_run", None)
    settings["campaign_fingerprint"] = _stable_fingerprint(fingerprint_payload)
    campaign_fingerprint = settings["campaign_fingerprint"]
    config_prefix = "strategy_tuning" if args.tuning_name == "strategy_tuning" else args.tuning_name
    _write_json(run_dir / f"{config_prefix}_config.json", settings)
    gpu_lock = threading.Lock()
    codegen_lock = threading.Lock()
    summaries: list[dict[str, Any]]
    worker_count = min(args.dataset_workers, len(datasets))
    if worker_count == 1:
        summaries = [
            _run_dataset(
                args=args,
                run_dir=run_dir,
                dataset=dataset,
                position=position,
                total_datasets=len(datasets),
                campaign_fingerprint=campaign_fingerprint,
                gpu_lock=gpu_lock,
                codegen_lock=codegen_lock,
            )
            for position, dataset in enumerate(datasets, start=1)
        ]
    else:
        results: dict[str, dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="dataset") as executor:
            future_to_dataset = {
                executor.submit(
                    _run_dataset,
                    args=args,
                    run_dir=run_dir,
                    dataset=dataset,
                    position=position,
                    total_datasets=len(datasets),
                    campaign_fingerprint=campaign_fingerprint,
                    gpu_lock=gpu_lock,
                    codegen_lock=codegen_lock,
                ): dataset
                for position, dataset in enumerate(datasets, start=1)
            }
            for future in as_completed(future_to_dataset):
                dataset = future_to_dataset[future]
                results[dataset] = future.result()
        summaries = [results[dataset] for dataset in datasets]
    _publish_global(run_dir, summaries, settings, args.tuning_name)
    failures = [
        item for item in summaries
        if item.get("status") in {"failed", "completed_with_failures"}
    ]
    unsupported = [item for item in summaries if item.get("status") == "unsupported"]
    print(
        f"complete: datasets={len(summaries)} failed={len(failures)} "
        f"unsupported={len(unsupported)} summary={run_dir / f'{config_prefix}_summary.json'}"
    )
    # A planning-only pass succeeds when it has faithfully classified every
    # artifact.  A real campaign must not silently claim success while any
    # fixed Agent-designed format remains unexecutable.
    return 1 if failures or (unsupported and not args.dry_run) else 0


if __name__ == "__main__":
    raise SystemExit(main())
