#!/usr/bin/env python3
"""Run only the format-design agent on a random sample of feature files."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import random
import sys
from typing import Any, Dict, List


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from sparsene.agent.config import resolve_model_config  # noqa: E402
from sparsene.op_gen.format_design_agent import (  # noqa: E402
    FormatDesignConfig,
    build_format_from_response,
    run_format_design_agent,
)


DEFAULT_FEATURE_DIR = Path("/mnt/lustre-client/huoda/selected_mtx")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _dataset_name(feature_path: Path) -> str:
    suffix = ".features.v3.json"
    name = feature_path.name
    return name[: -len(suffix)] if name.endswith(suffix) else feature_path.stem


def _render_basis_markdown(decision: Dict[str, Any]) -> str:
    evidence = decision.get("feature_evidence") or []
    tradeoffs = decision.get("tradeoffs") or []
    basis_cn = decision.get("decision_basis_cn") or []
    lines = [
        f"# {decision['matrix_name']} Format Design",
        "",
        f"- 模型: `{decision['model_name']}`",
        f"- 格式名称: `{decision.get('format_name')}`",
        f"- 格式家族: `{decision.get('format_family')}`",
        f"- Atomic: `{decision.get('atomic', {}).get('type')}`",
        f"- 参考 tile: `{decision.get('selected_candidate')}`",
        f"- DSL 校验: `{decision.get('dsl_validation', {}).get('status')}`",
        f"- 当前支持级别: `{decision.get('capability', {}).get('support_level')}`",
        f"- 置信度: `{decision.get('confidence')}`",
        f"- Feature 文件: `{decision['feature_file']}`",
        f"- Agent 原始日志: `{decision.get('agent_run_dir', '')}`",
        "",
        "## 中文设计依据",
        "",
        decision.get("design_summary") or "未返回设计摘要。",
        "",
        decision.get("rationale_cn") or "未返回中文 rationale。",
        "",
    ]
    if basis_cn:
        lines.extend(["## 中文要点", ""])
        lines.extend(f"- {item}" for item in basis_cn)
        lines.append("")
    if evidence:
        lines.extend(["## 特征证据", ""])
        for item in evidence:
            lines.append(
                f"- `{item.get('feature')}` = `{item.get('value')}`：{item.get('implication')}"
            )
        lines.append("")
    if tradeoffs:
        lines.extend(["## 权衡", ""])
        lines.extend(f"- {item}" for item in tradeoffs)
        lines.append("")
    alternatives = decision.get("alternatives_considered") or []
    if alternatives:
        lines.extend(["## 完整格式备选", ""])
        for item in alternatives:
            lines.append(
                f"- `{item.get('name')}`：{item.get('structure', '')}；"
                f"结论：{item.get('why_rejected', '')}"
            )
        lines.append("")
    requirements = decision.get("implementation_requirements") or []
    if requirements:
        lines.extend(["## 当前实现所需工作", ""])
        lines.extend(f"- {item}" for item in requirements)
        lines.append("")
    actions = decision.get("normalization_actions") or []
    if actions:
        lines.extend(["## 规范化动作", ""])
        lines.extend(f"- {item}" for item in actions)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Randomly sample matrix feature files and run the format-design agent only."
    )
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "logs" / "format_design_deepseek_v4pro",
    )
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--seed", type=str, default=None)
    parser.add_argument("--model", type=str, default="deepseek-v4-pro")
    parser.add_argument("--model-config", type=str, default=str(REPO_ROOT / "llm_models.yaml"))
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--timeout-sec", type=int, default=None)
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Optional explicit dataset names. If omitted, choose --count random datasets.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    feature_paths = sorted(args.feature_dir.glob("*.features.v3.json"))
    if not feature_paths:
        raise SystemExit(f"No feature files found in {args.feature_dir}")

    by_name = {_dataset_name(path): path for path in feature_paths}
    if args.datasets:
        missing = [name for name in args.datasets if name not in by_name]
        if missing:
            raise SystemExit(f"Unknown datasets: {', '.join(missing)}")
        selected = [by_name[name] for name in args.datasets]
        seed = "explicit"
    else:
        if args.count <= 0:
            raise SystemExit("--count must be positive")
        if args.count > len(feature_paths):
            raise SystemExit(f"--count={args.count} exceeds feature count={len(feature_paths)}")
        seed = args.seed or datetime.now().strftime("%Y%m%d%H%M%S")
        selected = random.Random(seed).sample(feature_paths, args.count)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_root / stamp
    run_dir.mkdir(parents=True, exist_ok=True)

    model_cfg = resolve_model_config(model_name=args.model, yaml_path=args.model_config)
    config = FormatDesignConfig(
        model_name=args.model,
        model=model_cfg.model,
        api_key=model_cfg.api_key,
        api_base_url=model_cfg.api_base_url,
        log_dir=str(run_dir / "agent_runs"),
        timeout_sec=args.timeout_sec or model_cfg.timeout_sec,
        temperature=args.temperature if args.temperature is not None else model_cfg.temperature,
        max_tokens=args.max_tokens or model_cfg.max_tokens,
        model_config_path=args.model_config,
    )

    summary: List[Dict[str, Any]] = []
    _write_json(
        run_dir / "selection.json",
        {
            "seed": seed,
            "model_name": args.model,
            "feature_dir": str(args.feature_dir),
            "selected_datasets": [_dataset_name(path) for path in selected],
        },
    )

    for feature_path in selected:
        matrix_name = _dataset_name(feature_path)
        matrix_dir = run_dir / matrix_name
        matrix_dir.mkdir(parents=True, exist_ok=True)
        features = _load_json(feature_path)
        response = run_format_design_agent(matrix_name, features, config)
        validation_status = "ok"
        validation_error = ""
        try:
            build_format_from_response(response)
        except Exception as exc:  # pragma: no cover - defensive run logging
            validation_status = "failed"
            validation_error = str(exc)

        decision = asdict(response)
        decision.update(
            {
                "matrix_name": matrix_name,
                "feature_file": str(feature_path),
                "model_name": args.model,
                "applied_candidate": None,
                "design_only": True,
                "validation_status": validation_status,
                "validation_error": validation_error,
                "agent_run_dir": response.agent_metadata.get("run_dir", ""),
            }
        )
        _write_json(matrix_dir / "format_decision.json", decision)
        (matrix_dir / "format_design_basis_zh.md").write_text(
            _render_basis_markdown(decision), encoding="utf-8"
        )
        summary.append(
            {
                "matrix_name": matrix_name,
                "format_name": response.format_name,
                "format_family": response.format_family,
                "atomic_type": response.atomic.get("type"),
                "selected_candidate": response.selected_candidate,
                "support_level": response.capability.get("support_level"),
                "dsl_status": response.dsl_validation.get("status"),
                "confidence": response.confidence,
                "validation_status": validation_status,
                "normalization_actions": response.normalization_actions,
                "format_decision": str(matrix_dir / "format_decision.json"),
                "basis_zh": str(matrix_dir / "format_design_basis_zh.md"),
                "agent_run_dir": response.agent_metadata.get("run_dir", ""),
            }
        )
        print(
            f"[format-design] {matrix_name}: format={response.format_name}/"
            f"{response.atomic.get('type', '?')} support={response.capability.get('support_level')} "
            f"confidence={response.confidence} validation={validation_status}"
        )

    _write_json(run_dir / "summary.json", {"run_dir": str(run_dir), "results": summary})
    print(f"[format-design] wrote summary: {run_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
