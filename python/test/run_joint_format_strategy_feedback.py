#!/usr/bin/env python3
"""Run nested Format-Agent and Strategy-Agent feedback for selected datasets.

For each dataset the outer loop proposes ``format_iterations`` complete Format
DSL designs.  Every accepted, executable format owns an inner strategy campaign
containing iteration 0 (the deterministic safe baseline) followed by
``strategy_agent_iterations`` NCU-guided Agent rounds.  The final result is the
fastest full-output-correct generated kernel across all outer/inner candidates.

The Format Agent receives compact format-level evidence from completed outer
iterations.  Raw NCU metrics remain private to the Strategy Agent because they
describe a fixed kernel implementation rather than the higher-level data
representation choice.
"""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
TEST_ROOT = PYTHON_ROOT / "test"
sys.path.insert(0, str(PYTHON_ROOT))
sys.path.insert(0, str(TEST_ROOT))

from run_format_design_batch import (  # noqa: E402
    _format_artifact,
    _is_retryable_transport_failure,
    _quality_issues,
    _render_basis_markdown,
)
from sparsene.agent.config import resolve_model_config  # noqa: E402
from sparsene.op_gen.backend_registry import select_backend  # noqa: E402
from sparsene.op_gen.early_stop import stop_decision  # noqa: E402
from sparsene.op_gen.format_design_agent import (  # noqa: E402
    DEFAULT_MAX_TOKENS,
    DESIGN_SCHEMA_VERSION,
    FormatDesignConfig,
    FormatDesignResponse,
    assess_backend_capability,
    run_format_design_agent,
    validate_format_dsl,
)
from sparsene.formats.SR_BCRS import SR_BCRS_FORMAT  # noqa: E402
from sparsene.formats.Acc_SpMM import BIT_TCF_FORMAT  # noqa: E402
from sparsene.utils.sparsene_reference import (  # noqa: E402
    DEFAULT_REFERENCE_CSV,
    SparseneReference,
    load_sparsene_reference,
)
from sparsene.performance_contract import (  # noqa: E402
    KERNEL_ONLY_EVALUATION_CONTRACT,
)
from sparsene.utils.incumbent_registry import (  # noqa: E402
    candidate_matches_scope,
    compact_incumbent_feedback,
    incumbent_as_joint_record,
    load_incumbent,
    update_registry_from_runs,
)
from sparsene.utils.temp_paths import sparsekg_toolchain_env  # noqa: E402


SCHEMA = "sparsene.joint_format_strategy_feedback.v1"
FORMAT_FEEDBACK_SCHEMA = "sparsene.format_outer_feedback.v1"
PERFORMANCE_CONTRACT = "sparsene.kernel_only_post_conversion.v1"
DEFAULT_FEATURE_DIR = Path("/mnt/lustre-client/huoda/selected_mtx")
DEFAULT_MODEL_CONFIG = REPO_ROOT / "llm_models.yaml"
DEFAULT_NCU = Path("/home/huoda/local/cuda/nsight-compute-2025.1.0/ncu")
DEFAULT_INCUMBENT_DIR = (
    REPO_ROOT / "logs" / "e2e_runs" / "incumbents" / "a100_a800_n512_fp32_atol1e-3_rtol1e-3"
)
FEATURE_SUFFIX = ".features.v3.json"
IMPLEMENTATION_PATHS = (
    "python/sparsene/op_gen/early_stop.py",
    "python/sparsene/op_gen/nvir/templates.py",
    "python/sparsene/op_gen/nvir/compiler_driver.py",
    "python/sparsene/op_gen/nvir/generate.py",
    "examples/src_fp32/acc/testbed/custom_format_e2e.cu",
    "python/test/run_joint_format_strategy_feedback.py",
    "python/test/run_v7_strategy_feedback.py",
    "python/test/generate_custom_format_operators.py",
    "python/test/run_custom_format_e2e.py",
    "python/sparsene/utils/temp_paths.py",
    "python/sparsene/op_gen/format_design_agent.py",
    "python/sparsene/op_gen/strategy_agent.py",
    "python/sparsene/op_gen/backend_registry.py",
    "python/sparsene/utils/incumbent_registry.py",
    "python/sparsene/utils/sparsene_reference.py",
    "python/test/update_e2e_incumbents.py",
    ".agent/skills/format-design/SKILL.md",
    ".agent/skills/strategy-optimization/SKILL.md",
    ".agent/skills/feedback-loop/SKILL.md",
    "scripts/run_joint_format_strategy_26datasets.sh",
)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_fingerprint(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def _git_value(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.stdout.strip()


def _append_event(run_dir: Path, event: dict[str, Any]) -> None:
    payload = {"time": datetime.now().isoformat(), **event}
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _dataset_name(path: Path) -> str:
    if path.name.endswith(FEATURE_SUFFIX):
        return path.name[: -len(FEATURE_SUFFIX)]
    return path.stem


def _select_features(feature_dir: Path, datasets: Iterable[str]) -> dict[str, Path]:
    paths = sorted(feature_dir.glob(f"*{FEATURE_SUFFIX}"))
    by_name = {_dataset_name(path): path for path in paths}
    requested = list(datasets)
    missing = [dataset for dataset in requested if dataset not in by_name]
    if missing:
        raise SystemExit(f"missing schema-v3 features for: {', '.join(missing)}")
    return {dataset: by_name[dataset] for dataset in requested}


def _format_signature(spec: dict[str, Any]) -> str:
    return _stable_fingerprint(
        {
            "levels": spec.get("levels", []),
            "atomic": spec.get("atomic", {}),
            "params": spec.get("params", {}),
        }
    )


def _fixed_sr_bcrs_response(
    reference: SparseneReference | None = None,
) -> FormatDesignResponse:
    """Construct a declared format0 matching the reference's storage semantics.

    The historical execution policy is recorded separately: strict load
    balancing and multi-binding are schedules, not different Format DSLs.
    """
    outer = BIT_TCF_FORMAT if reference and reference.atomic_type == "mco" else SR_BCRS_FORMAT
    inner = outer.child
    atomic_leaf = inner.child if inner is not None else None
    expected_atomic = reference.atomic_type if reference else "dense"
    if inner is None or atomic_leaf is None or atomic_leaf.atomic_type != expected_atomic:
        raise ValueError(f"declared format0 no longer has a two-level {expected_atomic} leaf")

    def axes(source: Any) -> list[dict[str, Any]]:
        result = []
        for axis in source:
            axis_type = ("s" if axis.is_sparse else "d") + ("v" if axis.is_varlen else "f")
            result.append({
                "name": axis.name,
                "type": axis_type,
                "direction": axis.direction.name,
                "length": str(axis.length),
                "reorder": axis.reorder,
            })
        return result

    levels = [
        {"level": 1, "name": "row_window_condensation", "axes": axes(outer.axes)},
        {"level": 2, "name": "tensorcore_block", "axes": axes(inner.axes)},
    ]
    atomic = {
        "type": expected_atomic,
        "x_axis": atomic_leaf.axes[0].name,
        "y_axis": atomic_leaf.axes[1].name,
        "layout": "row_major",
    }
    params = {"BLK_M": 16, "BLK_K": 8}
    name = f"{reference.variant if reference else 'SR-BCRS-16x8'}-fixed-format0"
    validation = validate_format_dsl(format_name=name, levels=levels, atomic=atomic, params=params)
    if validation.get("status") != "valid":
        raise ValueError(f"fixed SR-BCRS DSL invalid: {validation.get('errors')}")
    capability = assess_backend_capability(
        format_name=name, levels=levels, atomic=atomic, params=params,
        selected_candidate="16x8", validation=validation,
    )
    if not capability.get("execution_ready"):
        raise ValueError(f"fixed SR-BCRS is not executable: {capability.get('reasons')}")
    return FormatDesignResponse(
        request_id="fixed-sparsene-reference-format0" if reference else "fixed-sr-bcrs-format0",
        format_name=name,
        levels=levels,
        atomic=atomic,
        params=params,
        rationale="Deterministic format matching the historical Sparsene reference; not an Agent design.",
        confidence="high",
        format_family="tcf_mco" if expected_atomic == "mco" else "tcf_dense",
        design_summary=f"Declared column-condensed 16x8 {expected_atomic} Tensor-Core format.",
        selected_candidate="16x8",
        rationale_cn="按逐数据集历史最快 Sparsene 变体固定格式语义，不消耗格式智能体 token。",
        format_semantics_cn=[
            "将行分为16行窗口，对每个窗口的列做凝聚，再按8列分块，"
            + ("以 MCO 叶子保存块内非零元。" if expected_atomic == "mco"
               else "以 dense 叶子存储块内 A 值。"),
        ],
        selection_reason_cn="作为 format0 的可执行语义起点；历史耗时另列为参考，不冒充当前生成算子。",
        dsl_validation=validation,
        capability=capability,
        executable_profile=str(capability.get("matched_profile") or "none"),
        agent_metadata={"source": "fixed_repository_format", "model_called": False},
    )


def _run_fixed_initial_format(
    *, dataset: str, feature_path: Path, candidate_dir: Path,
    feedback: dict[str, Any], reference: SparseneReference | None = None,
) -> dict[str, Any]:
    response = _fixed_sr_bcrs_response(reference)
    origin = "historical_sparsene_best" if reference else "fixed_repository_sr_bcrs_16x8"
    dataset_dir = candidate_dir / "datasets" / dataset
    artifact_path = candidate_dir / "formats" / f"{dataset}.format.json"
    decision = asdict(response)
    decision.update({
        "dataset": dataset,
        "status": "accepted",
        "format_origin": origin,
        "historical_reference": reference.to_dict() if reference else None,
        "model_called": False,
        "attempt_count": 0,
        "total_token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "feature_file": str(feature_path.resolve()),
        "feature_sha256": _sha256(feature_path),
        "outer_feedback_contains_raw_ncu": False,
    })
    _write_json(dataset_dir / "matrix_features.json", _read_json(feature_path))
    _write_json(dataset_dir / "format_feedback_input.json", feedback)
    _write_json(dataset_dir / "format_decision.json", decision)
    spec = _format_artifact(dataset, response)
    spec["format_origin"] = origin
    if reference:
        spec["historical_reference"] = reference.to_dict()
    _write_json(artifact_path, spec)
    return {
        "status": "format_accepted",
        "format_origin": origin,
        "historical_reference": reference.to_dict() if reference else None,
        "format_artifact": str(artifact_path.resolve()),
        "format_artifact_sha256": _sha256(artifact_path),
        "format_signature": _format_signature(spec),
        "format": _compact_format(spec),
        "capability": response.capability,
    }


def _compact_levels(levels: Any) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    if not isinstance(levels, list):
        return compact
    for level in levels:
        if not isinstance(level, dict):
            continue
        axes = []
        for axis in level.get("axes", []):
            if not isinstance(axis, dict):
                continue
            axes.append(
                {
                    "name": axis.get("name"),
                    "type": axis.get("type"),
                    "direction": axis.get("direction"),
                    "length": axis.get("length"),
                    "reorder": axis.get("reorder"),
                }
            )
        compact.append(
            {"level": level.get("level"), "name": level.get("name"), "axes": axes}
        )
    return compact


def _compact_format(spec: dict[str, Any]) -> dict[str, Any]:
    atomic = spec.get("atomic", {}) if isinstance(spec.get("atomic"), dict) else {}
    return {
        "format_name": spec.get("format_name"),
        "format_family": spec.get("format_family"),
        "levels": _compact_levels(spec.get("levels")),
        "atomic": {
            "type": atomic.get("type"),
            "layout": atomic.get("layout"),
            "x_axis": atomic.get("x_axis"),
            "y_axis": atomic.get("y_axis"),
        },
        "params": spec.get("params", {}),
        "selected_candidate": spec.get("selected_candidate"),
    }


def _pick(mapping: Any, names: Iterable[str]) -> dict[str, Any]:
    if not isinstance(mapping, dict):
        return {}
    return {name: mapping.get(name) for name in names if name in mapping}


def _compact_best_result(entry: dict[str, Any] | None) -> dict[str, Any]:
    if not entry:
        return {}
    result = entry.get("e2e_result", {})
    timing = result.get("timing", {}) if isinstance(result, dict) else {}
    correctness = result.get("correctness", {}) if isinstance(result, dict) else {}
    packed = result.get("packed", {}) if isinstance(result, dict) else {}
    format_info = result.get("format", {}) if isinstance(result, dict) else {}
    return {
        "strategy_iteration": entry.get("iteration"),
        "timing": _pick(
            timing,
            (
                "generated_ms",
                "cusparse_ms",
                "speedup_vs_cusparse",
                "relative_spread_pct",
                "measurement_runs",
            ),
        ),
        "correctness": _pick(
            correctness,
            (
                "passed",
                "all_measurements_passed",
                "max_mismatches_across_measurements",
                "max_nonfinite_across_measurements",
            ),
        ),
        "physical_format": _pick(
            format_info,
            ("logical_blk_m", "physical_blk_m", "blk_k", "row_reorder", "packing_mode"),
        ),
        "kernel_visible_format_costs": _pick(
            packed,
            (
                "row_windows",
                "tiles",
                "aligned_values",
                "execution_segments",
                "max_tiles_per_execution_segment",
                "row_tiles_mean",
                "row_tiles_p95",
                "row_tiles_max",
                "distribution_heavy_tail",
                "execution_policy",
                "task_group_tiles",
                "task_groups",
                "tasks",
            ),
        ),
        "effective_strategy": {
            group: entry.get("strategy", {}).get(group, {})
            for group in ("tensor_placement", "pipeline", "compiler", "runtime_schedule")
            if isinstance(entry.get("strategy", {}).get(group), dict)
        },
    }


def _load_seed_feedback(seed_run_dir: Path | None, dataset: str) -> dict[str, Any]:
    if seed_run_dir is None:
        return {}
    root = seed_run_dir.resolve()
    payload: dict[str, Any] = {
        "source": "prior_audited_joint_campaign",
        "ncu_included": False,
    }
    artifact = root / "formats" / f"{dataset}.format.json"
    if artifact.is_file():
        spec = _read_json(artifact)
        payload["prior_format"] = _compact_format(spec)
        payload["prior_format_signature"] = _format_signature(spec)
        payload["prior_format_artifact_sha256"] = _sha256(artifact)
    comparison = root / "sparsene_2agent_speedup_comparison.csv"
    if comparison.is_file():
        with comparison.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                if row.get("dataset") != dataset:
                    continue
                payload["prior_normalized_performance"] = {
                    key: row.get(key)
                    for key in (
                        "sparsene_speedup",
                        "sparsene2_speedup",
                        "sparsene_2agent_speedup",
                        "two_agent_best_iteration",
                        "two_agent_generated_ms",
                        "two_agent_cusparse_ms",
                    )
                    if row.get(key) not in {None, ""}
                }
                payload["comparison_sha256"] = _sha256(comparison)
                break
    return payload


def _best_record(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    valid = [
        record
        for record in records
        if isinstance(record.get("best_generated_ms"), (int, float))
        and record.get("best_correctness_passed") is True
    ]
    return min(valid, key=lambda item: float(item["best_generated_ms"]), default=None)


def build_format_feedback(
    *,
    dataset: str,
    format_iteration: int,
    total_format_iterations: int,
    previous_records: list[dict[str, Any]],
    seed_feedback: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build high-level outer-loop feedback and deliberately omit NCU data."""

    history = []
    for record in previous_records:
        history.append(
            {
                "format_iteration": record.get("format_iteration"),
                "format_origin": record.get("format_origin", "format_agent"),
                "status": record.get("status"),
                "format_signature": record.get("format_signature"),
                "format": record.get("format"),
                "capability": record.get("capability"),
                "strategy_campaign": {
                    "candidate_budget": record.get("strategy_candidate_budget"),
                    "evaluated": record.get("strategy_candidates_evaluated"),
                    "validated": record.get("strategy_candidates_validated"),
                    "incorrect": record.get("strategy_candidates_incorrect"),
                    "failed": record.get("strategy_candidates_failed"),
                    "best_generated_ms": record.get("best_generated_ms"),
                    "best_speedup_vs_cusparse": record.get("best_speedup_vs_cusparse"),
                    "baseline_generated_ms": record.get("baseline_generated_ms"),
                    "improvement_vs_strategy_baseline": record.get(
                        "improvement_vs_strategy_baseline"
                    ),
                },
                "best_kernel_format_level_evidence": record.get("best_result", {}),
                "failure": record.get("error"),
            }
        )
    best = _best_record(previous_records)
    best_summary = None
    if best is not None:
        best_summary = {
            "format_iteration": best.get("format_iteration"),
            "format_origin": best.get("format_origin", "format_agent"),
            "format_signature": best.get("format_signature"),
            "best_generated_ms": best.get("best_generated_ms"),
            "best_speedup_vs_cusparse": best.get("best_speedup_vs_cusparse"),
            "format": best.get("format"),
        }
    return {
        "schema_version": FORMAT_FEEDBACK_SCHEMA,
        "dataset": dataset,
        "outer_iteration": format_iteration,
        "total_outer_iterations": total_format_iterations,
        "objective": PERFORMANCE_CONTRACT,
        "selected_format_execution_contract": {
            "required": True,
            "scope": "this joint SM80/A800 operator campaign only",
            "known_executable_starting_points": [
                "two-level row-window/column-condensed TCF with MCO or dense leaf",
                "one-level active fixed blocks with MCO or dense leaf",
                "three-level sparse-row/column-condensed SR-BCRS with dense leaf",
            ],
            "reliable_logical_tiles": {
                "BLK_M": [8, 16, 32],
                "BLK_K": [8, 16, 32],
            },
            "guidance": (
                "The selected DSL must pass the current executable backend and "
                "Format-to-RTS/codegen capability checks. Research-only formats may "
                "be discussed as alternatives, but cannot be the selected candidate "
                "in this campaign. Do not rename or silently canonicalize an unsupported "
                "proposal. Only use a reorder with an implemented host contract: "
                "row_rcm, first_col_sort, or jaccard_greedy; otherwise omit reorder. "
                "Use a supported hierarchy/atomic combination, and change the actual "
                "physical work rather than only the DSL name."
            ),
        },
        "role_boundary": {
            "format_agent_receives_raw_ncu": False,
            "reason": (
                "NCU diagnoses a fixed generated kernel and remains inside Strategy-Agent "
                "feedback; the Format Agent receives only representation-level outcomes."
            ),
            "format_agent_may_change": [
                "hierarchy",
                "axis_order_and_types",
                "atomic_leaf",
                "logical_tile_params",
                "reorder_semantics",
            ],
            "format_agent_must_not_change": [
                "strategy tensor placement",
                "pipeline wait/shift fields",
                "compiler register/launch bounds",
                "runtime task-group parameters",
            ],
        },
        "initial_calibration": seed_feedback or {},
        "previous_outer_iterations": history,
        "best_correct_format_so_far": best_summary,
        "decision_guidance": [
            "Compare the complete new representation against every previous outer iteration.",
            "Use measured kernel time, correctness, physical padding, active-tile count, metadata/restore semantics, and effective task distribution as format-level evidence.",
            "If inner strategy tuning barely improves a correct format, treat the remaining bottleneck as likely representation-dominated and change a format-level mechanism.",
            "If a previous format is unsupported or incorrect, do not claim performance for it; preserve the proposal in history and design a legal new hypothesis.",
            "Avoid an identical format signature unless retaining it is explicitly justified by evidence; do not express Strategy-Agent fields inside the Format DSL.",
            "Current backend capability is feasibility evidence, not a closed format menu or permission to silently replace a research proposal.",
        ],
    }


def _format_config(args: argparse.Namespace, log_dir: Path) -> FormatDesignConfig:
    model = resolve_model_config(
        model_name=args.format_model,
        temperature=args.format_temperature,
        max_tokens=args.format_max_tokens,
        timeout_sec=args.format_timeout_sec,
        yaml_path=str(args.model_config),
    )
    return FormatDesignConfig(
        model_name=args.format_model,
        model=model.model,
        api_key=model.api_key,
        api_base_url=model.api_base_url,
        log_dir=str(log_dir),
        timeout_sec=model.timeout_sec,
        temperature=model.temperature,
        max_tokens=model.max_tokens,
        reasoning_effort=args.format_reasoning_effort,
        model_config_path=str(args.model_config),
    )


def _format_retry_guidance(
    response: FormatDesignResponse, issues: list[str], attempt: int
) -> dict[str, Any]:
    """Give the next request actionable failure evidence, never a fake format."""
    fallback = response.agent_metadata.get("source") == "fallback"
    reason = str(response.rationale or "")
    truncated = response.agent_metadata.get("finish_reason") == "length"
    guidance: dict[str, Any] = {
        "failed_attempt": attempt,
        "blocking_issues": issues[:24],
        "previous_call_error": reason[:1000] if fallback else "",
        "previous_dsl": (
            {} if fallback else {
                "levels": response.levels,
                "atomic": response.atomic,
                "params": response.params,
            }
        ),
    }
    if truncated or "json parse failed" in reason.lower():
        guidance["response_budget"] = (
            "The previous output was truncated or not parseable. Return exactly "
            "one complete JSON object. Use concise strings and no markdown, "
            "duplicated explanations, long lists, or narrative outside JSON. "
            "Still provide every required field and at least three compact "
            "complete alternatives. Finish the JSON object before the token limit."
        )
    return guidance


def _joint_format_issues(response: FormatDesignResponse) -> list[str]:
    """Additional gate for a campaign that must execute its selected format."""
    issues = _quality_issues(response)
    if response.dsl_validation.get("status") != "valid":
        return issues
    spec = {
        "format_name": response.format_name,
        "levels": response.levels,
        "atomic": response.atomic,
        "params": response.params,
    }
    try:
        select_backend(spec, target="sm80")
    except (ValueError, KeyError, TypeError) as exc:
        issues.append(f"selected format has no executable SM80 backend: {exc}")
    if not response.capability.get("execution_ready"):
        issues.append(
            "selected format cannot complete the current Format-to-RTS/codegen "
            "capability check"
        )
    return issues


def _write_candidate_manifest(
    *,
    candidate_dir: Path,
    parent_run_dir: Path,
    dataset: str,
    format_iteration: int,
    feature_path: Path,
    feedback_path: Path,
) -> None:
    manifest = {
        "schema_version": SCHEMA,
        "run_id": candidate_dir.name,
        "phase": "joint_format_candidate",
        "parent_run_id": parent_run_dir.name,
        "dataset": dataset,
        "datasets": [dataset],
        "format_iteration": format_iteration,
        "format_design_schema_version": DESIGN_SCHEMA_VERSION,
        "evaluation_contract": KERNEL_ONLY_EVALUATION_CONTRACT,
        "feature_source": {
            "path": str(feature_path.resolve()),
            "sha256": _sha256(feature_path),
        },
        "format_feedback_input": {
            "path": str(feedback_path.resolve()),
            "sha256": _sha256(feedback_path),
            "contains_raw_ncu": False,
        },
    }
    _write_json(candidate_dir / "run_manifest.json", manifest)


def _run_format_agent(
    *,
    args: argparse.Namespace,
    dataset: str,
    features: dict[str, Any],
    feature_path: Path,
    candidate_dir: Path,
    feedback: dict[str, Any],
) -> tuple[FormatDesignResponse | None, dict[str, Any]]:
    dataset_dir = candidate_dir / "datasets" / dataset
    log_dir = dataset_dir / "format_agent"
    feedback_path = dataset_dir / "format_feedback_input.json"
    _write_json(feedback_path, feedback)
    _write_json(dataset_dir / "matrix_features.json", features)
    attempts: list[dict[str, Any]] = []
    old_decision_path = dataset_dir / "format_decision.json"
    if args.resume and old_decision_path.is_file():
        attempts = list(_read_json(old_decision_path).get("attempts") or [])
    attempt_base = max(
        [int(item.get("attempt") or 0) for item in attempts] + [0]
    )
    accepted: FormatDesignResponse | None = None
    last_response: FormatDesignResponse | None = None
    previous_issues: list[str] = list(
        (_read_json(old_decision_path).get("quality_issues") or [])
        if args.resume and old_decision_path.is_file() else []
    )
    quality_feedback: dict[str, Any] = {
        "campaign_requirement": (
            "This is an executable joint Format+Strategy campaign. The selected "
            "format must be executable by the current SM80 backend; research-only "
            "ideas belong in alternatives_considered, not the selected DSL."
        )
    }
    if previous_issues:
        quality_feedback.update({
            "blocking_issues": previous_issues,
            "previous_call_error": "Resumed after an unsuccessful format-design attempt.",
        })
    for attempt_idx in range(1, args.format_max_attempts + 1):
        attempt = attempt_base + attempt_idx
        attempt_feedback = deepcopy(feedback)
        if last_response is not None:
            attempt_feedback["validation_retry"] = {
                "attempt": attempt,
                "reason": "The previous call failed; return one new complete, legal JSON proposal.",
                "previous_quality_issues": previous_issues[:24],
                "previous_proposal": (
                    _compact_format(asdict(last_response))
                    if last_response.agent_metadata.get("source") != "fallback"
                    else {}
                ),
                "required_corrections": [
                    "Use atomic.layout exactly row_major or col_major.",
                    "For atomic.type=custom, provide non-empty value_arrays and metadata_arrays plus logical_mapping and restore_algorithm.",
                    "Preserve a genuinely new format hypothesis while satisfying every reported structural and evidence requirement.",
                ],
            }
        config = _format_config(args, log_dir)
        if attempt_idx > 1 and last_response is not None and (
            last_response.agent_metadata.get("finish_reason") == "length"
            or "json parse failed" in str(last_response.rationale or "").lower()
        ) and config.model.startswith("gpt-6"):
            # The OpenLux completion budget includes hidden reasoning. Reserve
            # it for the final JSON instead of repeating high-effort overflow.
            config.reasoning_effort = "low"
        response = run_format_design_agent(
            dataset,
            features,
            config,
            empirical_feedback=attempt_feedback,
            quality_feedback=quality_feedback,
            spmm_n=args.n,
        )
        last_response = response
        issues = _joint_format_issues(response)
        previous_issues = list(issues)
        attempt_record = {
            "attempt": attempt,
            "request_id": response.request_id,
            "agent_run_dir": response.agent_metadata.get("run_dir"),
            "finish_reason": response.agent_metadata.get("finish_reason"),
            "format_name": response.format_name,
            "format_family": response.format_family,
            "atomic_type": response.atomic.get("type"),
            "dsl_status": response.dsl_validation.get("status"),
            "support_level": response.capability.get("support_level"),
            "quality_issues": issues,
            "is_validation_retry": attempt > 1,
            "reasoning_effort_override": config.reasoning_effort,
            "token_usage": response.token_usage,
        }
        attempts.append(attempt_record)
        _write_json(dataset_dir / f"format_attempt_{attempt:02d}.json", attempt_record)
        if not issues:
            accepted = response
            break
        if attempt_idx < args.format_max_attempts:
            if response.agent_metadata.get("source") == "fallback" and not (
                _is_retryable_transport_failure(response)
                or response.agent_metadata.get("finish_reason") == "length"
                or "json parse failed" in str(response.rationale or "").lower()
            ):
                break  # e.g. authentication or unsupported request: fail fast
            quality_feedback = {
                **quality_feedback,
                **_format_retry_guidance(response, issues, attempt),
            }
            time.sleep(min(args.format_retry_delay_sec * 2 ** (attempt_idx - 1), 30.0))
    response = accepted or last_response
    if response is None:
        return None, {"status": "failed_format", "error": "no format response"}
    issues = _joint_format_issues(response)
    status = "accepted" if accepted is not None else "failed_format_quality_gate"
    decision = asdict(response)
    decision.update(
        {
            "dataset": dataset,
            "status": status,
            "quality_issues": issues,
            "attempt_count": len(attempts),
            "attempts": attempts,
            "total_token_usage": {
                field: sum(int((item.get("token_usage") or {}).get(field) or 0) for item in attempts)
                for field in ("prompt_tokens", "completion_tokens", "total_tokens", "reasoning_tokens")
            },
            "feature_file": str(feature_path.resolve()),
            "feature_sha256": _sha256(feature_path),
            "outer_feedback_path": str(feedback_path.resolve()),
            "outer_feedback_sha256": _sha256(feedback_path),
            "outer_feedback_contains_raw_ncu": False,
        }
    )
    _write_json(dataset_dir / "format_decision.json", decision)
    _write_json(dataset_dir / "raw_model_decision.json", response.raw_payload)
    _write_json(dataset_dir / "format_token_usage.json", response.token_usage)
    (dataset_dir / "format_design_basis_zh.md").write_text(
        _render_basis_markdown(dataset, decision), encoding="utf-8"
    )
    if accepted is None:
        return None, {
            "status": status,
            "error": "; ".join(issues),
            "format": _compact_format(decision),
            "capability": response.capability,
        }
    artifact_path = candidate_dir / "formats" / f"{dataset}.format.json"
    _write_json(artifact_path, _format_artifact(dataset, response))
    return response, {
        "status": "format_accepted",
        "format_artifact": str(artifact_path.resolve()),
        "format_artifact_sha256": _sha256(artifact_path),
        "format_signature": _format_signature(_read_json(artifact_path)),
        "format": _compact_format(_read_json(artifact_path)),
        "capability": response.capability,
    }


def _strategy_command(args: argparse.Namespace, candidate_dir: Path, dataset: str) -> list[str]:
    command = [
        sys.executable,
        str(TEST_ROOT / "run_v7_strategy_feedback.py"),
        str(candidate_dir),
        "--datasets",
        dataset,
        "--tuning-name",
        "strategy_tuning",
        "--agent-iterations",
        str(args.strategy_agent_iterations),
        "--n",
        str(args.n),
        "--warmup",
        str(args.warmup),
        "--repeat",
        str(args.repeat),
        "--measurement-runs",
        str(args.measurement_runs),
        "--jobs",
        "1",
        "--dataset-workers",
        "1",
        "--strategy-provider",
        args.strategy_provider,
        "--strategy-model",
        args.strategy_model,
        "--model-config",
        str(args.model_config),
        "--strategy-timeout-sec",
        str(args.strategy_timeout_sec),
        "--strategy-temperature",
        str(args.strategy_temperature),
        "--strategy-max-tokens",
        str(args.strategy_max_tokens),
        "--ncu-bin",
        str(args.ncu_bin),
        "--ncu-timeout-sec",
        str(args.ncu_timeout_sec),
    ]
    if args.skip_ncu:
        command.append("--skip-ncu")
    if getattr(args, "disable_early_stop", False):
        command.append("--disable-early-stop")
    if args.resume:
        command.append("--resume")
    return command


def _run_strategy_campaign(
    *, args: argparse.Namespace, candidate_dir: Path, dataset: str
) -> tuple[int, list[str]]:
    command = _strategy_command(args, candidate_dir, dataset)
    log_path = candidate_dir / "strategy_driver.log"
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(PYTHON_ROOT) + (os.pathsep + existing if existing else "")
    with log_path.open("a", encoding="utf-8") as log:
        log.write(
            f"\n[{datetime.now().isoformat()}] command="
            + json.dumps(command, ensure_ascii=False)
            + "\n"
        )
        log.flush()
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
        log.write(f"[{datetime.now().isoformat()}] returncode={result.returncode}\n")
    return result.returncode, command


def _record_from_candidate(
    *,
    args: argparse.Namespace,
    candidate_dir: Path,
    dataset: str,
    format_iteration: int,
    format_record: dict[str, Any],
    strategy_returncode: int | None,
) -> dict[str, Any]:
    record = {
        "schema_version": SCHEMA,
        "dataset": dataset,
        "format_iteration": format_iteration,
        "candidate_dir": str(candidate_dir.resolve()),
        "strategy_candidate_budget": args.strategy_agent_iterations + 1,
        "strategy_returncode": strategy_returncode,
        **format_record,
    }
    summary_path = candidate_dir / "datasets" / dataset / "strategy_tuning" / "summary.json"
    if not summary_path.is_file():
        record.setdefault("status", "failed_strategy")
        record.setdefault("error", "strategy campaign produced no dataset summary")
        return record
    summary = _read_json(summary_path)
    counts = summary.get("counts", {}) if isinstance(summary.get("counts"), dict) else {}
    best_path = candidate_dir / "datasets" / dataset / "strategy_tuning" / "best_iteration.json"
    best = _read_json(best_path) if best_path.is_file() else None
    record.update(
        {
            "status": summary.get("status", "unknown"),
            "strategy_summary_path": str(summary_path.resolve()),
            "strategy_candidates_evaluated": counts.get("total", 0),
            "strategy_candidates_validated": counts.get("validated", 0),
            "strategy_candidates_incorrect": counts.get("incorrect", 0),
            "strategy_candidates_failed": counts.get("failed", 0),
            "best_strategy_iteration": summary.get("best_iteration"),
            "best_generated_ms": summary.get("best_generated_ms"),
            "best_speedup_vs_cusparse": summary.get("best_speedup_vs_cusparse"),
            "baseline_generated_ms": summary.get("baseline_generated_ms"),
            "improvement_vs_strategy_baseline": summary.get("improvement_vs_baseline"),
            "best_operator_path": summary.get("best_operator_path"),
            "best_correctness_passed": bool(
                best
                and best.get("status") == "validated"
                and best.get("e2e_result", {}).get("correctness", {}).get("passed") is True
            ),
            "best_result": _compact_best_result(best),
        }
    )
    reference = format_record.get("historical_reference")
    if isinstance(reference, dict) and isinstance(summary.get("baseline_generated_ms"), (int, float)):
        record["generated_iteration_0_vs_historical_sparsene"] = (
            float(reference["historical_ms"]) / float(summary["baseline_generated_ms"])
        ) if summary["baseline_generated_ms"] > 0 else None
    if summary.get("error"):
        record["error"] = summary.get("error")
    return record


def _publish_dataset_summary(
    *,
    run_dir: Path,
    dataset: str,
    records: list[dict[str, Any]],
    args: argparse.Namespace,
    incumbent: dict[str, Any] | None = None,
) -> dict[str, Any]:
    dataset_dir = run_dir / "datasets" / dataset
    current_best = _best_record(records)
    incumbent_record = incumbent_as_joint_record(incumbent) if incumbent else None
    best = _best_record(
        [item for item in (current_best, incumbent_record) if item is not None]
    )
    candidate_budget = args.format_iterations * (args.strategy_agent_iterations + 1)
    evaluated = sum(
        int(record.get("strategy_candidates_evaluated") or 0) for record in records
    )
    failed_candidates = sum(
        int(record.get("strategy_candidates_failed") or 0) for record in records
    )
    reference = (
        load_sparsene_reference(dataset, args.sparsene_reference_csv)
        if getattr(args, "initial_format", None) == "sparsene_historical_best"
        else None
    )
    stop_path = dataset_dir / "early_stop_decision.json"
    outer_stopped = stop_path.is_file() and _read_json(stop_path).get("stop") is True
    # A smaller candidate count is complete only when an inner campaign really
    # stopped early. Unsupported/failed formats are gaps, never convergence.
    def inner_finished(record: dict[str, Any]) -> bool:
        if record.get("status") != "completed":
            return False
        if int(record.get("strategy_candidates_evaluated") or 0) >= args.strategy_agent_iterations + 1:
            return True
        summary_path = record.get("strategy_summary_path")
        if not summary_path:
            return False
        inner_stop = Path(summary_path).parent / "early_stop_decision.json"
        return inner_stop.is_file() and _read_json(inner_stop).get("stop") is True

    all_finished = failed_candidates == 0 and all(inner_finished(record) for record in records)
    if best and (outer_stopped or len(records) == args.format_iterations):
        status = "completed" if all_finished else "completed_with_gaps"
    else:
        status = "incomplete"
    summary = {
        "schema_version": SCHEMA,
        "dataset": dataset,
        "performance_contract": PERFORMANCE_CONTRACT,
        "historical_sparsene_reference": reference.to_dict() if reference else None,
        "historical_sparsene_variant": reference.variant if reference else None,
        "historical_sparsene_ms": reference.milliseconds if reference else None,
        "historical_sparsene_reference_speedup": 1.0 if reference else None,
        "historical_reference_is_generated_candidate": False,
        "trajectory_indexing": (
            "format0/iteration0=historical_reference; generated format0 heuristic is "
            "display iteration1 but internal strategy iteration0"
            if reference else "generated internal strategy iterations 0..K"
        ),
        "format_iterations_requested": args.format_iterations,
        "strategy_agent_iterations_per_format": args.strategy_agent_iterations,
        "early_stop": _read_json(stop_path) if stop_path.is_file() else None,
        "candidate_budget": candidate_budget,
        "format_iterations": records,
        "counts": {
            "format_iterations_recorded": len(records),
            "strategy_candidates_evaluated": evaluated,
            "strategy_candidates_validated": sum(
                int(record.get("strategy_candidates_validated") or 0) for record in records
            ),
            "strategy_candidates_incorrect": sum(
                int(record.get("strategy_candidates_incorrect") or 0) for record in records
            ),
            "strategy_candidates_failed": sum(
                int(record.get("strategy_candidates_failed") or 0) for record in records
            ),
        },
        "best_format_iteration": best.get("format_iteration") if best else None,
        "best_strategy_iteration": best.get("best_strategy_iteration") if best else None,
        "best_generated_ms": best.get("best_generated_ms") if best else None,
        "best_speedup_vs_cusparse": best.get("best_speedup_vs_cusparse") if best else None,
        "best_operator_path": best.get("best_operator_path") if best else None,
        "best_current_run_format_iteration": (
            current_best.get("format_iteration") if current_best else None
        ),
        "best_current_run_strategy_iteration": (
            current_best.get("best_strategy_iteration") if current_best else None
        ),
        "best_current_run_generated_ms": (
            current_best.get("best_generated_ms") if current_best else None
        ),
        "best_current_run_vs_historical_sparsene": (
            reference.milliseconds / float(current_best["best_generated_ms"])
            if reference and current_best and isinstance(current_best.get("best_generated_ms"), (int, float))
            and current_best["best_generated_ms"] > 0 else None
        ),
        "best_current_run_speedup_vs_cusparse": (
            current_best.get("best_speedup_vs_cusparse") if current_best else None
        ),
        "best_current_run_operator_path": (
            current_best.get("best_operator_path") if current_best else None
        ),
        "best_source": best.get("source_kind", "current_run") if best else None,
        "best_vs_historical_sparsene": (
            reference.milliseconds / float(best["best_generated_ms"])
            if reference and best and isinstance(best.get("best_generated_ms"), (int, float))
            and best["best_generated_ms"] > 0 else None
        ),
        "best_source_run": (
            best.get("source_run_id", run_dir.name) if best else None
        ),
        "incumbent_before_run": compact_incumbent_feedback(incumbent),
        "status": status,
    }
    _write_json(dataset_dir / "joint_summary.json", summary)
    if reference:
        trajectory_rows: list[dict[str, Any]] = [{
            "dataset": dataset,
            "format_iteration": 0,
            "display_iteration": 0,
            "internal_strategy_iteration": "",
            "point_type": "historical_sparsene_reference",
            "status": "historical_sentinel_only",
            "milliseconds": reference.milliseconds,
            "speedup_vs_historical_sparsene": 1.0,
            "variant": reference.variant,
            "operator_path": "",
        }]
        for record in records:
            format_index = int(record["format_iteration"])
            tuning_dir = Path(str(record["candidate_dir"])) / "datasets" / dataset / "strategy_tuning"
            for path in sorted(tuning_dir.glob("iteration_*/iteration_summary.json")):
                entry = _read_json(path)
                internal_index = entry.get("iteration")
                result = entry.get("e2e_result") or {}
                timing = result.get("timing") or {}
                milliseconds = timing.get("generated_ms")
                valid = (entry.get("status") == "validated"
                         and (result.get("correctness") or {}).get("passed") is True
                         and isinstance(milliseconds, (int, float)) and milliseconds > 0)
                trajectory_rows.append({
                    "dataset": dataset,
                    "format_iteration": format_index,
                    "display_iteration": (internal_index + 1 if format_index == 0 else internal_index)
                    if isinstance(internal_index, int) else "",
                    "internal_strategy_iteration": internal_index,
                    "point_type": "generated_kernel",
                    "status": entry.get("status"),
                    "milliseconds": milliseconds if valid else "",
                    "speedup_vs_historical_sparsene": (
                        reference.milliseconds / milliseconds if valid else ""
                    ),
                    "variant": (record.get("format") or {}).get("format_name", ""),
                    "operator_path": entry.get("operator_path", ""),
                })
        fields = list(trajectory_rows[0])
        with (dataset_dir / "performance_trajectory.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(trajectory_rows)
    if best:
        operator = Path(str(best["best_operator_path"]))
        artifact = Path(str(best["format_artifact"]))
        if operator.is_file():
            shutil.copy2(operator, dataset_dir / "best_operator.inc")
        if artifact.is_file():
            shutil.copy2(artifact, dataset_dir / "best_format.format.json")
        _write_json(dataset_dir / "best_joint_candidate.json", best)
    return summary


def _publish_global(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    summaries = []
    for dataset in args.datasets:
        path = run_dir / "datasets" / dataset / "joint_summary.json"
        if path.is_file():
            summaries.append(_read_json(path))
    payload = {
        "schema_version": SCHEMA,
        "performance_contract": PERFORMANCE_CONTRACT,
        "settings": {
            "format_iterations": args.format_iterations,
            "strategy_agent_iterations": args.strategy_agent_iterations,
            "strategy_candidates_per_format": args.strategy_agent_iterations + 1,
            "total_candidate_budget_per_dataset": args.format_iterations
            * (args.strategy_agent_iterations + 1),
            "format_feedback_contains_raw_ncu": False,
            "n": args.n,
            "dtype": "fp32",
            "mma": "tf32",
            "warmup": args.warmup,
            "repeat": args.repeat,
            "measurement_runs": args.measurement_runs,
        },
        "counts": {
            "datasets_requested": len(args.datasets),
            "datasets_recorded": len(summaries),
            "datasets_completed": sum(item.get("status") == "completed" for item in summaries),
            "datasets_with_valid_best": sum(
                isinstance(item.get("best_generated_ms"), (int, float))
                for item in summaries
            ),
            "datasets_retaining_incumbent": sum(
                item.get("best_source") == "incumbent_registry" for item in summaries
            ),
            "datasets_with_current_run_best": sum(
                isinstance(item.get("best_current_run_generated_ms"), (int, float))
                for item in summaries
            ),
        },
        "results": summaries,
    }
    _write_json(run_dir / "joint_feedback_summary.json", payload)
    fields = [
        "dataset",
        "status",
        "historical_sparsene_variant",
        "historical_sparsene_ms",
        "historical_sparsene_reference_speedup",
        "historical_reference_is_generated_candidate",
        "candidate_budget",
        "best_format_iteration",
        "best_strategy_iteration",
        "best_generated_ms",
        "best_vs_historical_sparsene",
        "best_speedup_vs_cusparse",
        "best_source",
        "best_source_run",
        "best_operator_path",
        "best_current_run_format_iteration",
        "best_current_run_strategy_iteration",
        "best_current_run_generated_ms",
        "best_current_run_vs_historical_sparsene",
        "best_current_run_speedup_vs_cusparse",
        "best_current_run_operator_path",
    ]
    with (run_dir / "joint_feedback_summary.csv").open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summaries)
    return payload


def _existing_records(run_dir: Path, dataset: str) -> list[dict[str, Any]]:
    records = []
    for path in sorted((run_dir / "datasets" / dataset / "format_iterations").glob("format_*/joint_candidate_summary.json")):
        records.append(_read_json(path))
    return records


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--format-iterations", type=int, default=6, help="maximum format candidates; early stopping may finish sooner")
    parser.add_argument(
        "--initial-format",
        choices=("agent", "sr_bcrs_16x8", "sparsene_historical_best"),
        default="agent",
        help="Use the per-dataset historical fastest Sparsene format, fixed SR-BCRS, or an Agent design as format 0.",
    )
    parser.add_argument(
        "--sparsene-reference-csv", type=Path, default=DEFAULT_REFERENCE_CSV,
        help="Audited A100 N=512 FP32 per-dataset historical fastest variant catalog.",
    )
    parser.add_argument("--strategy-agent-iterations", type=int, default=5)
    parser.add_argument("--disable-early-stop", action="store_true", help="run all format and strategy iterations")
    parser.add_argument("--initial-feedback-run-dir", type=Path)
    parser.add_argument(
        "--incumbent-dir",
        type=Path,
        default=DEFAULT_INCUMBENT_DIR,
        help=(
            "cross-run correctness-gated incumbent registry used as the dataset-level "
            "performance floor and updated by strictly faster correct candidates"
        ),
    )
    parser.add_argument(
        "--skip-incumbent-update",
        action="store_true",
        help="read the incumbent performance floor but do not update the registry",
    )
    parser.add_argument("--format-model", default="deepseek-v4-pro")
    parser.add_argument("--format-temperature", type=float, default=0.25)
    parser.add_argument("--format-max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--format-timeout-sec", type=int, default=600)
    parser.add_argument("--format-max-attempts", type=int, default=1)
    parser.add_argument(
        "--format-reasoning-effort",
        choices=("low", "medium", "high", "xhigh", "max"),
        default=None,
        help="Optional format-only reasoning override; retries after GPT-6 truncation use low.",
    )
    parser.add_argument("--format-retry-delay-sec", type=float, default=1.0)
    parser.add_argument(
        "--abort-on-format-failure", action="store_true",
        help="Stop at a failed outer format rather than count it as a completed design; resume in the same directory after inspection.",
    )
    parser.add_argument("--strategy-provider", default="openai_http")
    parser.add_argument("--strategy-model", default="deepseek-v4-pro")
    parser.add_argument("--strategy-temperature", type=float, default=0.2)
    parser.add_argument("--strategy-max-tokens", type=int, default=16384)
    parser.add_argument("--strategy-timeout-sec", type=int, default=300)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=32)
    parser.add_argument("--measurement-runs", type=int, default=3)
    parser.add_argument("--ncu-bin", type=Path, default=DEFAULT_NCU)
    parser.add_argument("--ncu-timeout-sec", type=int, default=900)
    parser.add_argument("--skip-ncu", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="write and validate the 3x6 plan without Agent, compiler, or GPU calls",
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.format_iterations < 1:
        raise SystemExit("--format-iterations must be positive")
    if args.strategy_agent_iterations < 0:
        raise SystemExit("--strategy-agent-iterations must be nonnegative")
    if args.format_max_attempts < 1:
        raise SystemExit("--format-max-attempts must be positive")
    if args.measurement_runs < 1:
        raise SystemExit("--measurement-runs must be positive")
    if (args.n, args.warmup, args.repeat) != (512, 3, 32):
        raise SystemExit("joint campaign is fixed to N=512, warmup=3, repeat=32")
    if len(set(args.datasets)) != len(args.datasets):
        raise SystemExit("--datasets contains duplicates")
    if not args.model_config.is_file():
        raise SystemExit(f"missing model config: {args.model_config}")
    if args.initial_format == "sparsene_historical_best":
        for dataset in args.datasets:
            load_sparsene_reference(dataset, args.sparsene_reference_csv)
    if args.initial_feedback_run_dir is not None and not args.initial_feedback_run_dir.is_dir():
        raise SystemExit(
            f"missing initial feedback run directory: {args.initial_feedback_run_dir}"
        )
    if not args.skip_ncu and not args.ncu_bin.is_file():
        raise SystemExit(f"missing NCU executable: {args.ncu_bin}")


def _write_root_manifest(args: argparse.Namespace, run_dir: Path, features: dict[str, Path]) -> None:
    model = resolve_model_config(
        model_name=args.format_model,
        temperature=args.format_temperature,
        max_tokens=args.format_max_tokens,
        timeout_sec=args.format_timeout_sec,
        yaml_path=str(args.model_config),
    )
    manifest = {
        "schema_version": SCHEMA,
        "run_id": run_dir.name,
        "created_at": datetime.now().isoformat(),
        "phase": "nested_format_strategy_feedback",
        "datasets": args.datasets,
        "dataset_count": len(args.datasets),
        "evaluation_contract": KERNEL_ONLY_EVALUATION_CONTRACT,
        "loop": {
            "format_iterations": args.format_iterations,
            "initial_format": args.initial_format,
            "sparsene_reference_csv_sha256": (
                _sha256(args.sparsene_reference_csv)
                if args.initial_format == "sparsene_historical_best" else None
            ),
            "strategy_agent_iterations_per_format": args.strategy_agent_iterations,
            "early_stop_enabled": not args.disable_early_stop,
            "heuristic_baseline_per_format": 1,
            "strategy_candidates_per_format": args.strategy_agent_iterations + 1,
            "candidate_budget_per_dataset": args.format_iterations
            * (args.strategy_agent_iterations + 1),
            "selection": (
                "minimum median generated_ms among the incoming registry incumbent and "
                "all full-output-correct candidates"
            ),
        },
        "feedback_boundary": {
            "strategy_agent": "correctness + repeated timing + NCU + effective runtime mapping",
            "format_agent": "format-level aggregate only; no raw NCU",
        },
        "models": {
            "format": {
                **model.to_dict(),
                "effective_max_tokens": model.max_tokens,
                "max_attempts": args.format_max_attempts,
                "reasoning_effort_override": args.format_reasoning_effort,
            },
            "strategy": {
                "model_name": args.strategy_model,
                "provider": args.strategy_provider,
                "model_config_path": str(args.model_config.resolve()),
                "temperature": args.strategy_temperature,
                "max_tokens": args.strategy_max_tokens,
                "timeout_sec": args.strategy_timeout_sec,
            },
        },
        "protocol": {
            "n": args.n,
            "dtype": "fp32",
            "mma": "tf32",
            "warmup": args.warmup,
            "repeat": args.repeat,
            "measurement_runs": args.measurement_runs,
            "ncu_enabled": not args.skip_ncu,
            "serial_gpu_execution": True,
        },
        "features": {
            dataset: {"path": str(path.resolve()), "sha256": _sha256(path)}
            for dataset, path in features.items()
        },
        "initial_feedback_run": (
            str(args.initial_feedback_run_dir.resolve())
            if args.initial_feedback_run_dir
            else None
        ),
        "incumbent_registry": {
            "path": str(args.incumbent_dir.resolve()),
            "read_enabled": True,
            "update_enabled": not args.skip_incumbent_update,
            "registry_sha256_at_start": (
                _sha256(args.incumbent_dir / "registry.json")
                if (args.incumbent_dir / "registry.json").is_file()
                else None
            ),
        },
        "environment": {
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "virtual_env": os.environ.get("VIRTUAL_ENV", ""),
        },
        "repository": {
            "root": str(REPO_ROOT),
            "revision": _git_value("rev-parse", "HEAD"),
            "dirty": bool(_git_value("status", "--short")),
            "implementation_files": {
                relative: _sha256(REPO_ROOT / relative) for relative in IMPLEMENTATION_PATHS
            },
        },
        "layout": {
            "per_dataset": "datasets/<dataset>/",
            "per_format": f"datasets/<dataset>/format_iterations/format_<00..{args.format_iterations - 1:02d}>/",
            "per_strategy": f".../datasets/<dataset>/strategy_tuning/iteration_<00..{args.strategy_agent_iterations:02d}>/",
            "dataset_best": "datasets/<dataset>/best_joint_candidate.json",
            "cross_run_incumbent": str(args.incumbent_dir.resolve()),
        },
    }
    manifest_path = run_dir / "run_manifest.json"
    if args.resume and manifest_path.is_file():
        existing = _read_json(manifest_path)
        immutable = ("datasets", "loop", "evaluation_contract", "protocol")
        mismatch = []
        for name in immutable:
            prior = deepcopy(existing.get(name))
            current = manifest.get(name)
            if name == "loop" and isinstance(prior, dict):
                # Older runs had only Agent-designed format 0.
                prior.setdefault("initial_format", "agent")
                prior.setdefault("sparsene_reference_csv_sha256", None)
            if prior != current:
                mismatch.append(name)
        def model_identity(payload: dict[str, Any]) -> dict[str, Any]:
            models = deepcopy(payload.get("models") or {})
            format_model = models.get("format") or {}
            format_model.pop("api_key_present", None)
            # Older manifests predate this optional override; absent and None
            # have the same request semantics, so preserve their resume path.
            format_model.setdefault("reasoning_effort_override", None)
            format_model.setdefault("reasoning_effort", None)
            format_model.setdefault("thinking_mode", None)
            format_model.setdefault("stream", False)
            return models
        if model_identity(existing) != model_identity(manifest):
            mismatch.append("models")
        if mismatch:
            raise SystemExit(
                "resume manifest mismatch for immutable fields: " + ", ".join(mismatch)
            )
        return
    if manifest_path.exists():
        raise SystemExit(f"run manifest already exists; use --resume: {manifest_path}")
    _write_json(manifest_path, manifest)


def main() -> int:
    args = _parse_args()
    _validate_args(args)
    toolchain_env = sparsekg_toolchain_env()
    for key in ("TMPDIR", "TMP", "TEMP"):
        os.environ[key] = toolchain_env[key]
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    feature_paths = _select_features(args.feature_dir.resolve(), args.datasets)
    _write_root_manifest(args, run_dir, feature_paths)
    plan = {
        "schema_version": SCHEMA,
        "datasets": args.datasets,
        "format_iterations": args.format_iterations,
        "initial_format": args.initial_format,
        "historical_reference_is_generated_candidate": False,
        "strategy_iterations_per_format": list(range(args.strategy_agent_iterations + 1)),
        "candidate_budget_per_dataset": args.format_iterations
        * (args.strategy_agent_iterations + 1),
        "format_feedback_contains_raw_ncu": False,
    }
    _write_json(run_dir / "execution_plan.json", plan)
    if args.dry_run:
        print(
            f"planned datasets={len(args.datasets)} formats={args.format_iterations} "
            f"candidates_per_dataset={plan['candidate_budget_per_dataset']} run={run_dir}"
        )
        return 0

    failures = 0
    for dataset_position, dataset in enumerate(args.datasets, start=1):
        dataset_dir = run_dir / "datasets" / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)
        feature_path = feature_paths[dataset]
        features = _read_json(feature_path)
        records = _existing_records(run_dir, dataset) if args.resume else []
        incumbent = load_incumbent(args.incumbent_dir, dataset)
        if incumbent is not None:
            compatible, issues = candidate_matches_scope(incumbent)
            if not compatible:
                raise SystemExit(
                    f"incompatible incumbent for {dataset}: {', '.join(issues)}"
                )
        seed = dict(_load_seed_feedback(args.initial_feedback_run_dir, dataset) or {})
        seed["registry_incumbent"] = compact_incumbent_feedback(incumbent)
        reference = (
            load_sparsene_reference(dataset, args.sparsene_reference_csv)
            if args.initial_format == "sparsene_historical_best" else None
        )
        if reference:
            seed["historical_sparsene_reference"] = reference.to_dict()
        print(
            f"[{dataset_position:02d}/{len(args.datasets):02d}] {dataset}: "
            f"resume_outer={len(records)}",
            flush=True,
        )
        for format_iteration in range(args.format_iterations):
            if format_iteration and not args.disable_early_stop:
                observations = [{"iteration": item.get("format_iteration"),
                    "status": "validated" if item.get("best_correctness_passed") is True and item.get("status") == "completed" else "invalid",
                    "time_ms": item.get("best_generated_ms"),
                    "spread_pct": ((item.get("best_result") or {}).get("timing") or {}).get("relative_spread_pct")}
                    for item in records if int(item.get("format_iteration", -1)) < format_iteration]
                decision_stop = stop_decision(observations)
                decision_stop.update({"scope": "format", "next_iteration": format_iteration})
                _write_json(dataset_dir / "early_stop_decision.json", decision_stop)
                _write_json(dataset_dir / "early_stop" / f"before_{format_iteration:02d}.json", decision_stop)
                if decision_stop["stop"]:
                    _append_event(run_dir, {"event": "format_early_stop", "dataset": dataset, **decision_stop})
                    break
            candidate_dir = (
                dataset_dir / "format_iterations" / f"format_{format_iteration:02d}"
            )
            candidate_summary_path = candidate_dir / "joint_candidate_summary.json"
            existing = next(
                (
                    item
                    for item in records
                    if item.get("format_iteration") == format_iteration
                ),
                None,
            )
            if args.resume and existing is not None and existing.get("status") in {
                "completed",
                "unsupported",
                "incorrect",
            }:
                print(
                    f"  format {format_iteration}: reused terminal {existing.get('status')}",
                    flush=True,
                )
                continue
            feedback = build_format_feedback(
                dataset=dataset,
                format_iteration=format_iteration,
                total_format_iterations=args.format_iterations,
                previous_records=[
                    item for item in records if int(item.get("format_iteration", -1)) < format_iteration
                ],
                seed_feedback=seed,
            )
            candidate_dir.mkdir(parents=True, exist_ok=True)
            feedback_path = candidate_dir / "datasets" / dataset / "format_feedback_input.json"
            _write_json(feedback_path, feedback)
            # The fixed format and resumed artifacts must satisfy the same
            # feature-snapshot contract as fresh Agent designs.
            _write_json(candidate_dir / "datasets" / dataset / "matrix_features.json", features)
            _write_candidate_manifest(
                candidate_dir=candidate_dir,
                parent_run_dir=run_dir,
                dataset=dataset,
                format_iteration=format_iteration,
                feature_path=feature_path,
                feedback_path=feedback_path,
            )
            _append_event(
                run_dir,
                {
                    "event": "format_iteration_started",
                    "dataset": dataset,
                    "format_iteration": format_iteration,
                },
            )
            artifact_path = candidate_dir / "formats" / f"{dataset}.format.json"
            decision_path = candidate_dir / "datasets" / dataset / "format_decision.json"
            format_record: dict[str, Any]
            if args.resume and artifact_path.is_file() and decision_path.is_file():
                spec = _read_json(artifact_path)
                decision = _read_json(decision_path)
                format_record = {
                    "status": "format_accepted",
                    "format_origin": spec.get("format_origin", "format_agent"),
                    "format_artifact": str(artifact_path.resolve()),
                    "format_artifact_sha256": _sha256(artifact_path),
                    "format_signature": _format_signature(spec),
                    "format": _compact_format(spec),
                    "capability": decision.get("capability", {}),
                    "historical_reference": spec.get("historical_reference"),
                }
            elif format_iteration == 0 and args.initial_format in {
                "sr_bcrs_16x8", "sparsene_historical_best"
            }:
                format_record = _run_fixed_initial_format(
                    dataset=dataset,
                    feature_path=feature_path,
                    candidate_dir=candidate_dir,
                    feedback=feedback,
                    reference=reference,
                )
            else:
                _response, format_record = _run_format_agent(
                    args=args,
                    dataset=dataset,
                    features=features,
                    feature_path=feature_path,
                    candidate_dir=candidate_dir,
                    feedback=feedback,
                )
            if format_record.get("status") != "format_accepted":
                record = {
                    "schema_version": SCHEMA,
                    "dataset": dataset,
                    "format_iteration": format_iteration,
                    "candidate_dir": str(candidate_dir.resolve()),
                    "strategy_candidate_budget": args.strategy_agent_iterations + 1,
                    **format_record,
                }
                _write_json(candidate_summary_path, record)
                if args.abort_on_format_failure:
                    _append_event(
                        run_dir,
                        {
                            "event": "campaign_paused_for_format_failure",
                            "dataset": dataset,
                            "format_iteration": format_iteration,
                            "error": record.get("error"),
                        },
                    )
                    print(
                        f"format {format_iteration} failed after {args.format_max_attempts} "
                        f"attempts; inspect {decision_path} and resume this run",
                        flush=True,
                    )
                    return 2
            else:
                returncode, command = _run_strategy_campaign(
                    args=args, candidate_dir=candidate_dir, dataset=dataset
                )
                _write_json(
                    candidate_dir / "strategy_driver_command.json",
                    {"argv": command, "returncode": returncode},
                )
                record = _record_from_candidate(
                    args=args,
                    candidate_dir=candidate_dir,
                    dataset=dataset,
                    format_iteration=format_iteration,
                    format_record=format_record,
                    strategy_returncode=returncode,
                )
                _write_json(candidate_summary_path, record)
            records = [
                item for item in records if item.get("format_iteration") != format_iteration
            ] + [record]
            records.sort(key=lambda item: int(item.get("format_iteration", -1)))
            _publish_dataset_summary(
                run_dir=run_dir,
                dataset=dataset,
                records=records,
                args=args,
                incumbent=incumbent,
            )
            _publish_global(run_dir, args)
            _append_event(
                run_dir,
                {
                    "event": "format_iteration_completed",
                    "dataset": dataset,
                    "format_iteration": format_iteration,
                    "status": record.get("status"),
                    "best_generated_ms": record.get("best_generated_ms"),
                },
            )
            print(
                f"  format {format_iteration}: status={record.get('status')} "
                f"best_ms={record.get('best_generated_ms')}",
                flush=True,
            )
        dataset_summary = _publish_dataset_summary(
            run_dir=run_dir,
            dataset=dataset,
            records=records,
            args=args,
            incumbent=incumbent,
        )
        if dataset_summary.get("status") != "completed":
            failures += 1
        _publish_global(run_dir, args)
        if not args.skip_incumbent_update and _best_record(records) is not None:
            registry_report = update_registry_from_runs(
                args.incumbent_dir,
                [run_dir],
                datasets=[dataset],
            )
            _append_event(
                run_dir,
                {
                    "event": "incumbent_registry_updated",
                    "dataset": dataset,
                    "registry": str(args.incumbent_dir.resolve()),
                    "inserted": registry_report["inserted"],
                    "replaced": registry_report["replaced"],
                    "retained": registry_report["retained"],
                    "rejected": registry_report["rejected"],
                },
            )
    global_summary = _publish_global(run_dir, args)
    print(
        f"complete datasets={len(args.datasets)} failures={failures} "
        f"summary={run_dir / 'joint_feedback_summary.json'}"
    )
    return 1 if failures or global_summary["counts"]["datasets_completed"] != len(args.datasets) else 0


if __name__ == "__main__":
    raise SystemExit(main())
