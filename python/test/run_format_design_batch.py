#!/usr/bin/env python3
"""Run audited DeepSeek format design over persistent matrix features.

The run directory is intentionally phase-oriented so later strategy search,
code generation, compilation, correctness, timing, and NCU artifacts can be
appended to the same per-dataset directory.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Dict, Iterable, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

from sparsene.agent.config import resolve_model_config  # noqa: E402
from sparsene.op_gen.format_design_agent import (  # noqa: E402
    DEFAULT_MAX_TOKENS,
    DESIGN_SCHEMA_VERSION,
    FormatDesignConfig,
    FormatDesignResponse,
    build_format_from_response,
    run_format_design_agent,
)
from sparsene.performance_contract import (  # noqa: E402
    EVALUATION_CONTRACT_VERSION,
    KERNEL_ONLY_EVALUATION_CONTRACT,
)
from sparsene.transform.rts import derive_rts  # noqa: E402


DEFAULT_FEATURE_DIR = Path("/mnt/lustre-client/huoda/selected_mtx")
DEFAULT_RUN_ROOT = REPO_ROOT / "logs" / "e2e_runs"
FEATURE_SUFFIX = ".features.v3.json"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dataset_name(path: Path) -> str:
    return path.name[: -len(FEATURE_SUFFIX)] if path.name.endswith(FEATURE_SUFFIX) else path.stem


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


def _append_event(run_dir: Path, event: Dict[str, Any]) -> None:
    payload = {"time": datetime.now().isoformat(), **event}
    with (run_dir / "events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _quality_issues(response: FormatDesignResponse) -> List[str]:
    issues: List[str] = []
    if response.agent_metadata.get("source") == "fallback" or not response.agent_metadata.get("run_dir"):
        issues.append("agent fallback was used")
    if response.dsl_validation.get("status") != "valid":
        issues.extend(
            f"invalid DSL: {error}" for error in response.dsl_validation.get("errors", [])
        )
    if len(response.feature_evidence) < 6:
        issues.append("fewer than six grounded feature evidence records")
    if not response.design_summary.strip():
        issues.append("missing design_summary")
    if not response.format_family.strip():
        issues.append("missing format_family")
    if not response.rationale_cn.strip():
        issues.append("missing rationale_cn")
    if len(response.format_semantics_cn) < 2:
        issues.append("incomplete format_semantics_cn")
    if len(response.alternatives_considered) < 3:
        issues.append("fewer than three complete-format alternatives")
    for index, alternative in enumerate(response.alternatives_considered, start=1):
        required = ("name", "structure", "pros", "cons", "why_rejected")
        if not all(alternative.get(key) for key in required):
            issues.append(f"alternative {index} is incomplete")
    if len(response.transformation_plan) < 2:
        issues.append("transformation_plan is too short")
    for index, transformation in enumerate(response.transformation_plan, start=1):
        if not all(
            transformation.get(key) for key in ("op", "target", "result", "reason")
        ):
            issues.append(f"transformation_plan entry {index} is incomplete")
    for level in response.levels:
        for axis in level.get("axes", []):
            if not str(axis.get("purpose", "")).strip():
                issues.append(
                    f"missing purpose for axis {level.get('name', '?')}.{axis.get('name', '?')}"
                )
    if not str(response.atomic.get("rationale", "")).strip():
        issues.append("missing atomic rationale")
    if not response.selection_reason_cn.strip():
        issues.append("missing selection_reason_cn")
    if not response.implementation_requirements:
        issues.append("missing implementation_requirements")
    if not response.risks_cn:
        issues.append("missing risks_cn")
    try:
        fmt = build_format_from_response(response)
        if response.capability.get("rts_derivable") and not derive_rts(fmt).sequence:
            issues.append("empty RTS")
    except Exception as exc:
        issues.append(f"format validation failed: {exc}")
    return issues


def _render_basis_markdown(dataset: str, decision: Dict[str, Any]) -> str:
    resolved_model = decision.get("agent_metadata", {}).get("resolved_model", {})
    lines = [
        f"# {dataset} 格式设计",
        "",
        f"- 模型：`{resolved_model.get('model_name', 'fallback')}`",
        f"- 设计 schema：`{decision['schema_version']}`",
        f"- 格式名称：`{decision['format_name']}`",
        f"- 格式家族：`{decision['format_family']}`",
        f"- Atomic：`{decision.get('atomic', {}).get('type', 'unknown')}`",
        f"- 参考 tile：`{decision['selected_candidate']}`",
        f"- 当前代码支持级别：`{decision.get('capability', {}).get('support_level', 'unknown')}`",
        f"- 匹配的已知 profile：`{decision.get('capability', {}).get('matched_profile') or 'none'}`",
        f"- DSL 校验：`{decision.get('dsl_validation', {}).get('status', 'unknown')}`",
        f"- 置信度：`{decision['confidence']}`",
        f"- 输入 tokens：`{decision['token_usage']['prompt_tokens']}`",
        f"- 输出 tokens：`{decision['token_usage']['completion_tokens']}`",
        f"- Agent 对话目录：`{decision['agent_metadata']['run_dir']}`",
        "",
        "## 性能评价与计时边界",
        "",
        f"- 契约：`{decision.get('evaluation_contract', {}).get('version', 'missing')}`",
        f"- 主指标：{decision.get('evaluation_contract', {}).get('timed_scope_cn', '')}",
        f"- 排名规则：{decision.get('evaluation_contract', {}).get('ranking_rule_cn', '')}",
        f"- 工程记录：{decision.get('evaluation_contract', {}).get('engineering_reporting_cn', '')}",
        "- 主指标明确排除：",
    ]
    lines.extend(
        f"  - {item}"
        for item in decision.get("evaluation_contract", {}).get(
            "excluded_from_primary_metric", []
        )
    )
    lines.extend([
        "",
        "## 设计摘要",
        "",
        decision.get("design_summary", ""),
        "",
        "## 格式语义理解",
        "",
    ])
    lines.extend(f"- {item}" for item in decision.get("format_semantics_cn", []))
    lines.extend(["", "## Format DSL", "", "```json"])
    lines.append(
        json.dumps(
            {
                "levels": decision.get("levels", []),
                "atomic": decision.get("atomic", {}),
                "params": decision.get("params", {}),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    lines.extend(["```", "", "## 变换计划", ""])
    for item in decision.get("transformation_plan", []):
        lines.append(
            f"- `{item.get('op', '?')}` `{item.get('target', '?')}`："
            f"{item.get('result', '')}；{item.get('reason', '')}"
        )
    lines.extend(["", "## 完整格式备选", ""])
    for item in decision.get("alternatives_considered", []):
        lines.append(f"### {item.get('name', 'unnamed')}")
        lines.append("")
        lines.append(f"- 结构：{item.get('structure', '')}")
        lines.extend(f"- 优点：{value}" for value in item.get("pros", []))
        lines.extend(f"- 缺点：{value}" for value in item.get("cons", []))
        lines.append(f"- 结论：{item.get('why_rejected', '')}")
        lines.append("")
    lines.extend(
        [
            "## 最终选择原因",
            "",
            decision.get("selection_reason_cn", ""),
            "",
            "## 总体设计依据",
            "",
            decision.get("rationale_cn", ""),
            "",
        ]
    )
    if decision.get("decision_basis_cn"):
        lines.extend(f"- {item}" for item in decision["decision_basis_cn"])
        lines.append("")
    lines.extend(["## 精确特征证据", ""])
    for item in decision.get("feature_evidence", []):
        lines.append(
            f"- `{item.get('feature')}` = `{item.get('value')}`：{item.get('implication')}"
        )
    lines.extend(["", "## 权衡与风险", ""])
    lines.extend(f"- {item}" for item in decision.get("tradeoffs", []))
    lines.extend(f"- {item}" for item in decision.get("risks_cn", []))
    lines.extend(["", "## 当前实现所需工作", ""])
    lines.extend(f"- {item}" for item in decision.get("implementation_requirements", []))
    capability = decision.get("capability", {})
    if capability.get("reasons"):
        lines.extend(["", "## 确定性 capability 审查", ""])
        lines.extend(f"- {item}" for item in capability["reasons"])
    if decision.get("normalization_actions"):
        lines.extend(["", "## 确定性规范化动作", ""])
        lines.extend(f"- {item}" for item in decision["normalization_actions"])
    return "\n".join(lines).rstrip() + "\n"


def _format_artifact(dataset: str, response: FormatDesignResponse) -> Dict[str, Any]:
    return {
        "dataset": dataset,
        "schema_version": response.schema_version,
        "design_profile": response.design_profile,
        "format_name": response.format_name,
        "format_family": response.format_family,
        "design_summary": response.design_summary,
        "executable_profile": response.executable_profile,
        "selected_candidate": response.selected_candidate,
        "levels": response.levels,
        "atomic": response.atomic,
        "params": response.params,
        "transformation_plan": response.transformation_plan,
        "feature_evidence": response.feature_evidence,
        "format_semantics_cn": response.format_semantics_cn,
        "alternatives_considered": response.alternatives_considered,
        "selection_reason_cn": response.selection_reason_cn,
        "rationale_cn": response.rationale_cn,
        "implementation_requirements": response.implementation_requirements,
        "risks_cn": response.risks_cn,
        "confidence": response.confidence,
        "dsl_validation": response.dsl_validation,
        "capability": response.capability,
        "evaluation_contract": response.evaluation_contract,
        "normalization_actions": response.normalization_actions,
        "request_id": response.request_id,
    }


def _collect_token_usage(run_dir: Path) -> Dict[str, Any]:
    def sum_usages(paths: Iterable[Path]) -> Dict[str, int]:
        totals = {
            "request_count": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "reasoning_tokens": 0,
            "total_tokens": 0,
        }
        for path in paths:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                usage = payload.get("usage", payload)
                totals["request_count"] += 1
                totals["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
                totals["completion_tokens"] += int(usage.get("completion_tokens") or 0)
                totals["reasoning_tokens"] += int(usage.get("reasoning_tokens") or 0)
                totals["total_tokens"] += int(
                    usage.get("total_tokens")
                    or (int(usage.get("prompt_tokens") or 0) + int(usage.get("completion_tokens") or 0))
                )
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                continue
        return totals

    format_usage = sum_usages(run_dir.glob("datasets/*/format_agent/*/audit.json"))
    connectivity_usage = sum_usages(
        run_dir.glob("connectivity/*_model_connectivity/*/usage.json")
    )
    combined = {
        key: format_usage[key] + connectivity_usage[key]
        for key in format_usage
    }
    return {
        "format_design": format_usage,
        "connectivity": connectivity_usage,
        "all_recorded_requests": combined,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--model", default="deepseek-v4-pro")
    parser.add_argument("--model-config", default=str(REPO_ROOT / "llm_models.yaml"))
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--timeout-sec", type=int, default=None)
    parser.add_argument("--max-attempts", type=int, default=1)
    parser.add_argument("--retry-delay-sec", type=float, default=1.0)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse accepted per-dataset results already present in --run-dir.",
    )
    return parser.parse_args()


def _select_features(feature_dir: Path, datasets: Iterable[str] | None) -> List[Path]:
    paths = sorted(feature_dir.glob(f"*{FEATURE_SUFFIX}"))
    by_name = {_dataset_name(path): path for path in paths}
    if datasets:
        requested = list(datasets)
        missing = [name for name in requested if name not in by_name]
        if missing:
            raise SystemExit(f"Unknown datasets: {', '.join(missing)}")
        return [by_name[name] for name in requested]
    return paths


def main() -> int:
    args = _parse_args()
    if args.max_attempts < 1:
        raise SystemExit("--max-attempts must be at least 1")
    feature_paths = _select_features(args.feature_dir, args.datasets)
    if not feature_paths:
        raise SystemExit(f"No {FEATURE_SUFFIX} files found in {args.feature_dir}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = (args.run_dir or args.run_root / f"{stamp}_{args.model}_format_design").resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "format_design_summary.json").exists() and not args.resume:
        raise SystemExit(f"Format design already completed in {run_dir}")
    manifest_path = run_dir / "run_manifest.json"
    if args.resume and manifest_path.exists():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        existing_schema = existing_manifest.get("format_design_schema_version")
        if existing_schema != DESIGN_SCHEMA_VERSION:
            raise SystemExit(
                "Refusing to mix format-design schemas in one run directory: "
                f"existing={existing_schema or 'pre-versioned'}, current={DESIGN_SCHEMA_VERSION}. "
                "Start a new run directory for the current schema instead of mixing contracts."
            )
        existing_evaluation = existing_manifest.get("evaluation_contract", {}).get("version")
        if existing_evaluation != EVALUATION_CONTRACT_VERSION:
            raise SystemExit(
                "Refusing to mix performance-evaluation contracts in one run directory: "
                f"existing={existing_evaluation or 'missing'}, "
                f"current={EVALUATION_CONTRACT_VERSION}. Start a new run directory."
            )
    (run_dir / "datasets").mkdir(exist_ok=True)
    (run_dir / "formats").mkdir(exist_ok=True)

    model = resolve_model_config(model_name=args.model, yaml_path=args.model_config)
    manifest = {
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "created_at": datetime.now().isoformat(),
        "phase": "format_design",
        "format_design_schema_version": DESIGN_SCHEMA_VERSION,
        "evaluation_contract": KERNEL_ONLY_EVALUATION_CONTRACT,
        "future_phases": ["strategy_search", "codegen", "build", "correctness", "performance", "ncu"],
        "model": model.to_dict(),
        "format_agent_options": {
            "json_output": True,
            "effective_max_tokens": (
                args.max_tokens if args.max_tokens is not None else max(model.max_tokens, DEFAULT_MAX_TOKENS)
            ),
            "max_attempts": args.max_attempts,
        },
        "feature_dir": str(args.feature_dir.resolve()),
        "datasets": [_dataset_name(path) for path in feature_paths],
        "dataset_count": len(feature_paths),
        "environment": {
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "virtual_env": os.environ.get("VIRTUAL_ENV", ""),
            "required_virtual_env": "/mnt/lustre-client/huoda/venv/llm-sog",
        },
        "repository": {
            "root": str(REPO_ROOT),
            "revision": _git_value("rev-parse", "HEAD"),
            "dirty": bool(_git_value("status", "--short")),
        },
        "layout": {
            "connectivity": "connectivity/",
            "per_dataset": "datasets/<dataset>/",
            "final_formats": "formats/<dataset>.format.json",
            "future_strategy": "datasets/<dataset>/strategy/",
            "future_codegen": "datasets/<dataset>/codegen/",
            "future_build_and_measurement": "datasets/<dataset>/iterations/",
        },
    }
    if not args.resume or not manifest_path.exists():
        _write_json(manifest_path, manifest)
    _append_event(
        run_dir,
        {
            "event": "format_design_resumed" if args.resume else "format_design_started",
            "dataset_count": len(feature_paths),
        },
    )

    config_base = {
        "model_name": args.model,
        "model": model.model,
        "api_key": model.api_key,
        "api_base_url": model.api_base_url,
        "timeout_sec": args.timeout_sec or model.timeout_sec,
        "temperature": args.temperature if args.temperature is not None else model.temperature,
        "max_tokens": (
            args.max_tokens if args.max_tokens is not None else max(model.max_tokens, DEFAULT_MAX_TOKENS)
        ),
        "model_config_path": args.model_config,
    }

    results: List[Dict[str, Any]] = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_reasoning_tokens = 0
    failures = 0

    for index, feature_path in enumerate(feature_paths, start=1):
        dataset = _dataset_name(feature_path)
        dataset_dir = run_dir / "datasets" / dataset
        agent_log_dir = dataset_dir / "format_agent"
        existing_decision_path = dataset_dir / "format_decision.json"
        existing_format_path = run_dir / "formats" / f"{dataset}.format.json"
        if args.resume and existing_decision_path.exists() and existing_format_path.exists():
            existing = json.loads(existing_decision_path.read_text(encoding="utf-8"))
            if existing.get("status") == "ok":
                usage = existing.get("token_usage", {})
                attempts = existing.get("attempts", [])
                row = {
                    "dataset": dataset,
                    "status": "ok",
                    "format_name": existing.get("format_name", ""),
                    "format_family": existing.get("format_family", ""),
                    "atomic_type": existing.get("atomic", {}).get("type", ""),
                    "reference_tile": existing.get("selected_candidate", ""),
                    "support_level": existing.get("capability", {}).get("support_level", ""),
                    "matched_profile": existing.get("capability", {}).get("matched_profile") or "",
                    "dsl_status": existing.get("dsl_validation", {}).get("status", ""),
                    "confidence": existing.get("confidence", ""),
                    "finish_reason": existing.get("agent_metadata", {}).get("finish_reason") or "",
                    "prompt_tokens": int(usage.get("prompt_tokens") or 0),
                    "completion_tokens": int(usage.get("completion_tokens") or 0),
                    "reasoning_tokens": int(usage.get("reasoning_tokens") or 0),
                    "all_attempt_prompt_tokens": sum(
                        int(item.get("token_usage", {}).get("prompt_tokens") or 0)
                        for item in attempts
                    ),
                    "all_attempt_completion_tokens": sum(
                        int(item.get("token_usage", {}).get("completion_tokens") or 0)
                        for item in attempts
                    ),
                    "attempts": len(attempts) or int(existing.get("attempt_count") or 1),
                    "normalization_count": len(existing.get("normalization_actions", [])),
                    "decision_path": str(existing_decision_path),
                    "format_path": str(existing_format_path),
                    "agent_run_dir": existing.get("agent_metadata", {}).get("run_dir", ""),
                }
                results.append(row)
                print(
                    f"[{index:02d}/{len(feature_paths):02d}] {dataset}: reused accepted result",
                    flush=True,
                )
                continue
        dataset_dir.mkdir(parents=True, exist_ok=True)
        features = json.loads(feature_path.read_text(encoding="utf-8"))
        _write_json(dataset_dir / "matrix_features.json", features)
        _write_json(
            dataset_dir / "input_provenance.json",
            {
                "source_feature_file": str(feature_path.resolve()),
                "source_sha256": _sha256(feature_path),
                "schema_version": features.get("schema_version"),
            },
        )
        _append_event(run_dir, {"event": "dataset_started", "dataset": dataset, "index": index})

        accepted: FormatDesignResponse | None = None
        attempt_records: List[Dict[str, Any]] = []
        last_response: FormatDesignResponse | None = None
        for attempt in range(1, args.max_attempts + 1):
            config = FormatDesignConfig(log_dir=str(agent_log_dir), **config_base)
            response = run_format_design_agent(dataset, features, config)
            last_response = response
            attempt_usage = response.token_usage
            total_prompt_tokens += int(attempt_usage.get("prompt_tokens") or 0)
            total_completion_tokens += int(attempt_usage.get("completion_tokens") or 0)
            total_reasoning_tokens += int(attempt_usage.get("reasoning_tokens") or 0)
            issues = _quality_issues(response)
            attempt_record = {
                "attempt": attempt,
                "request_id": response.request_id,
                "agent_run_dir": response.agent_metadata.get("run_dir", ""),
                "token_usage": response.token_usage,
                "format_name": response.format_name,
                "format_family": response.format_family,
                "atomic_type": response.atomic.get("type"),
                "reference_tile": response.selected_candidate,
                "support_level": response.capability.get("support_level"),
                "dsl_status": response.dsl_validation.get("status"),
                "confidence": response.confidence,
                "finish_reason": response.agent_metadata.get("finish_reason"),
                "normalization_actions": response.normalization_actions,
                "quality_issues": issues,
            }
            attempt_records.append(attempt_record)
            _write_json(dataset_dir / f"attempt_{attempt:02d}.json", attempt_record)
            if not issues:
                accepted = response
                break
            _append_event(
                run_dir,
                {"event": "format_attempt_rejected", "dataset": dataset, **attempt_record},
            )
            if attempt < args.max_attempts:
                time.sleep(args.retry_delay_sec)

        response = accepted or last_response
        if response is None:  # pragma: no cover - loop always produces a response or raises
            raise RuntimeError(f"No response produced for {dataset}")
        issues = _quality_issues(response)
        status = "ok" if accepted is not None else "failed_quality_gate"
        if status != "ok":
            failures += 1

        decision = asdict(response)
        decision.update(
            {
                "dataset": dataset,
                "status": status,
                "quality_issues": issues,
                "attempt_count": len(attempt_records),
                "attempts": attempt_records,
                "feature_file": str(feature_path.resolve()),
                "feature_sha256": _sha256(feature_path),
                "applied_candidate": None,
                "design_only": True,
            }
        )
        _write_json(dataset_dir / "format_decision.json", decision)
        _write_json(dataset_dir / "raw_model_decision.json", response.raw_payload)
        _write_json(dataset_dir / "token_usage.json", response.token_usage)
        (dataset_dir / "format_design_basis_zh.md").write_text(
            _render_basis_markdown(dataset, decision), encoding="utf-8"
        )
        format_path = run_dir / "formats" / f"{dataset}.format.json"
        if status == "ok":
            _write_json(format_path, _format_artifact(dataset, response))

        usage = response.token_usage
        all_attempt_prompt_tokens = sum(
            int(item["token_usage"].get("prompt_tokens") or 0) for item in attempt_records
        )
        all_attempt_completion_tokens = sum(
            int(item["token_usage"].get("completion_tokens") or 0) for item in attempt_records
        )
        row = {
            "dataset": dataset,
            "status": status,
            "format_name": response.format_name,
            "format_family": response.format_family,
            "atomic_type": response.atomic.get("type", ""),
            "reference_tile": response.selected_candidate,
            "support_level": response.capability.get("support_level", ""),
            "matched_profile": response.capability.get("matched_profile") or "",
            "dsl_status": response.dsl_validation.get("status", ""),
            "confidence": response.confidence,
            "finish_reason": response.agent_metadata.get("finish_reason") or "",
            "prompt_tokens": int(usage.get("prompt_tokens") or 0),
            "completion_tokens": int(usage.get("completion_tokens") or 0),
            "reasoning_tokens": int(usage.get("reasoning_tokens") or 0),
            "all_attempt_prompt_tokens": all_attempt_prompt_tokens,
            "all_attempt_completion_tokens": all_attempt_completion_tokens,
            "attempts": len(attempt_records),
            "normalization_count": len(response.normalization_actions),
            "decision_path": str(dataset_dir / "format_decision.json"),
            "format_path": str(format_path) if status == "ok" else "",
            "agent_run_dir": response.agent_metadata.get("run_dir", ""),
        }
        results.append(row)
        _append_event(run_dir, {"event": "dataset_completed", **row})
        print(
            f"[{index:02d}/{len(feature_paths):02d}] {dataset}: status={status} "
            f"format={response.format_name}/{response.atomic.get('type', '?')} "
            f"support={response.capability.get('support_level')} "
            f"tokens={row['prompt_tokens']}+{row['completion_tokens']} "
            f"attempts={row['attempts']}",
            flush=True,
        )

    token_totals = _collect_token_usage(run_dir)
    summary = {
        "run_dir": str(run_dir),
        "format_design_schema_version": DESIGN_SCHEMA_VERSION,
        "model": model.to_dict(),
        "dataset_count": len(feature_paths),
        "ok": len(feature_paths) - failures,
        "failed": failures,
        "format_family_counts": {
            value: sum(1 for row in results if row["status"] == "ok" and row["format_family"] == value)
            for value in sorted({row["format_family"] for row in results})
        },
        "atomic_type_counts": {
            value: sum(1 for row in results if row["status"] == "ok" and row["atomic_type"] == value)
            for value in sorted({row["atomic_type"] for row in results})
        },
        "support_level_counts": {
            value: sum(1 for row in results if row["status"] == "ok" and row["support_level"] == value)
            for value in sorted({row["support_level"] for row in results})
        },
        "reference_tile_counts": {
            value: sum(1 for row in results if row["status"] == "ok" and row["reference_tile"] == value)
            for value in sorted({row["reference_tile"] for row in results})
        },
        "token_totals": token_totals,
        "results": results,
    }
    _write_json(run_dir / "format_design_summary.json", summary)
    _write_json(run_dir / "token_usage_total.json", token_totals)
    with (run_dir / "format_design_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    (args.run_root.resolve() / "latest_run.txt").parent.mkdir(parents=True, exist_ok=True)
    (args.run_root.resolve() / "latest_run.txt").write_text(str(run_dir) + "\n", encoding="utf-8")
    _append_event(run_dir, {"event": "format_design_completed", "ok": len(feature_paths) - failures, "failed": failures, "token_totals": token_totals})
    print(f"Format design complete: ok={len(feature_paths) - failures} failed={failures}")
    print(f"Run directory: {run_dir}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
