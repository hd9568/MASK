"""Maintain a correctness-gated cross-run SpMM incumbent registry.

The registry is deliberately independent from one experiment directory.  Each
dataset keeps immutable versioned copies of the selected generated operator,
Format DSL artifact, normalized strategy, and generation manifest.  A small
``current.json`` pointer is atomically replaced only when a compatible,
full-output-correct candidate has a lower warmed kernel time.
"""

from __future__ import annotations

import csv
from datetime import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from typing import Any, Iterable


SCHEMA = "sparsekg.e2e_incumbent_registry.v1"
PERFORMANCE_CONTRACT = "sparsene.kernel_only_post_conversion.v1"
DEFAULT_PROTOCOL: dict[str, Any] = {
    "n": 512,
    "dtype": "fp32",
    "mma": "tf32",
    "warmup": 3,
    "repeat": 32,
    "measurement_runs": 3,
    "ranking_statistic": "median",
}
DEFAULT_HARDWARE_SCOPE: dict[str, Any] = {
    "target": "sm80",
    "compute_capability": 80,
    "sm_count": 108,
    "comparison_scope": "A100/A800-class SM80 single GPU",
}
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SPARSENE_PAPER_BASELINE = (
    REPO_ROOT
    / "sota_results"
    / "summary"
    / "paper_a100_n512_fp32_sparsene_per_dataset.csv"
)


class IncumbentRegistryError(ValueError):
    """Raised when a candidate cannot satisfy the registry contract."""


def _now() -> str:
    return datetime.now().astimezone().isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise IncumbentRegistryError(f"expected JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_write_json(path: Path, payload: Any) -> None:
    _atomic_write_text(
        path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    )


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def _redact_secrets(value: Any) -> Any:
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if key.lower() in {"api_key", "authorization", "access_token"}:
                result[key] = "<redacted>"
            else:
                result[key] = _redact_secrets(item)
        return result
    if isinstance(value, list):
        return [_redact_secrets(item) for item in value]
    return value


def _resolve_path(value: Any, run_dir: Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else (run_dir / path).resolve()


def _infer_run_sm_count(run_dir: Path) -> tuple[int | None, str | None]:
    """Recover a run-level SM count when the selected iteration predates NCU capture."""

    for path in sorted(run_dir.glob("datasets/**/ncu/ncu_summary.json")):
        try:
            payload = _read_json(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        value = payload.get("sm_count")
        if isinstance(value, (int, float)) and int(value) > 0:
            return int(value), str(path.resolve())
    return None, None


def _require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise IncumbentRegistryError(f"missing {label}: {path}")
    return path


def _require_close(actual: float, expected: float, label: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-9):
        raise IncumbentRegistryError(
            f"{label} mismatch: iteration={actual} summary={expected}"
        )


def _candidate_from_iteration(
    *,
    run_dir: Path,
    dataset: str,
    operator_path: Path,
    source_kind: str,
    source_summary_path: Path,
    format_iteration: int | None = None,
    format_path: Path | None = None,
    expected_generated_ms: float | None = None,
    expected_cusparse_ms: float | None = None,
    expected_speedup: float | None = None,
) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    operator_path = _require_file(operator_path.resolve(), "generated operator")
    iteration_dir = operator_path.parent.parent
    iteration_summary_path = _require_file(
        iteration_dir / "iteration_summary.json", "iteration summary"
    )
    iteration = _read_json(iteration_summary_path)
    if iteration.get("dataset") != dataset:
        raise IncumbentRegistryError(
            f"dataset mismatch in {iteration_summary_path}: {iteration.get('dataset')}"
        )
    if iteration.get("status") != "validated":
        raise IncumbentRegistryError(
            f"candidate is not validated: {dataset} status={iteration.get('status')}"
        )
    if iteration.get("performance_contract") != PERFORMANCE_CONTRACT:
        raise IncumbentRegistryError(
            f"performance contract mismatch for {dataset}: "
            f"{iteration.get('performance_contract')}"
        )

    result = iteration.get("e2e_result")
    if not isinstance(result, dict):
        raise IncumbentRegistryError(f"missing e2e_result for {dataset}")
    correctness = result.get("correctness")
    timing = result.get("timing")
    if not isinstance(correctness, dict) or not isinstance(timing, dict):
        raise IncumbentRegistryError(f"missing correctness/timing for {dataset}")
    if correctness.get("passed") is not True:
        raise IncumbentRegistryError(f"correctness did not pass for {dataset}")
    measurement_runs = int(timing.get("measurement_runs") or 0)
    if measurement_runs > 1 and correctness.get("all_measurements_passed") is not True:
        raise IncumbentRegistryError(
            f"not all independent measurements passed for {dataset}"
        )
    if timing.get("contract") != PERFORMANCE_CONTRACT:
        raise IncumbentRegistryError(f"timing contract mismatch for {dataset}")

    generated_ms = float(timing.get("generated_ms") or 0.0)
    cusparse_ms = float(timing.get("cusparse_ms") or 0.0)
    speedup = float(timing.get("speedup_vs_cusparse") or 0.0)
    if min(generated_ms, cusparse_ms, speedup) <= 0.0:
        raise IncumbentRegistryError(f"non-positive timing for {dataset}")
    if expected_generated_ms is not None:
        _require_close(generated_ms, expected_generated_ms, "generated_ms")
    if expected_cusparse_ms is not None:
        _require_close(cusparse_ms, expected_cusparse_ms, "cusparse_ms")
    if expected_speedup is not None:
        _require_close(speedup, expected_speedup, "speedup_vs_cusparse")

    generation_manifest_path = _require_file(
        _resolve_path(iteration.get("generation_manifest"), run_dir),
        "generation manifest",
    )
    generation_manifest = _read_json(generation_manifest_path)
    if generation_manifest.get("status") != "generated":
        raise IncumbentRegistryError(f"operator was not freshly generated for {dataset}")
    manifest_operator_sha = generation_manifest.get("operator_sha256")
    operator_sha = _sha256(operator_path)
    if manifest_operator_sha and manifest_operator_sha != operator_sha:
        raise IncumbentRegistryError(f"operator SHA256 mismatch for {dataset}")

    if format_path is None:
        source = generation_manifest.get("format_source", {})
        if not isinstance(source, dict) or not source.get("artifact_path"):
            raise IncumbentRegistryError(f"missing format source for {dataset}")
        format_path = _resolve_path(source["artifact_path"], run_dir)
    format_path = _require_file(format_path.resolve(), "Format DSL artifact")
    format_spec = _read_json(format_path)
    format_sha = _sha256(format_path)
    manifest_format_sha = (
        generation_manifest.get("format_source", {}).get("artifact_sha256")
        if isinstance(generation_manifest.get("format_source"), dict)
        else None
    )
    if manifest_format_sha and manifest_format_sha != format_sha:
        raise IncumbentRegistryError(f"format SHA256 mismatch for {dataset}")

    shape = result.get("shape", {}) if isinstance(result.get("shape"), dict) else {}
    backend = (
        result.get("backend_plan", {})
        if isinstance(result.get("backend_plan"), dict)
        else {}
    )
    ncu = (
        iteration.get("ncu_profile", {})
        if isinstance(iteration.get("ncu_profile"), dict)
        else {}
    )
    protocol = {
        "n": int(shape.get("N") or 0),
        "dtype": "fp32",
        "mma": "tf32",
        "warmup": int(timing.get("warmup") or 0),
        "repeat": int(timing.get("repeat") or 0),
        "measurement_runs": measurement_runs,
        "ranking_statistic": timing.get("ranking_statistic"),
    }
    sm_count = ncu.get("sm_count")
    sm_count_source = "selected_iteration_ncu"
    sm_count_evidence_path = None
    if not isinstance(sm_count, (int, float)) or int(sm_count) <= 0:
        sm_count, sm_count_evidence_path = _infer_run_sm_count(run_dir)
        sm_count_source = "same_run_ncu_inference" if sm_count is not None else None
    hardware = {
        "target": backend.get("target"),
        "compute_capability": backend.get("compute_capability"),
        "sm_count": int(sm_count) if isinstance(sm_count, (int, float)) else None,
        "sm_count_source": sm_count_source,
        "sm_count_evidence_path": sm_count_evidence_path,
    }
    strategy = (
        iteration.get("strategy", {})
        if isinstance(iteration.get("strategy"), dict)
        else {}
    )
    atomic = format_spec.get("atomic", {})
    if not isinstance(atomic, dict):
        atomic = {}
    packed = result.get("packed", {}) if isinstance(result.get("packed"), dict) else {}
    physical_format = (
        result.get("format", {}) if isinstance(result.get("format"), dict) else {}
    )
    source_summary_path = _require_file(source_summary_path.resolve(), "source summary")
    candidate_id = _fingerprint(
        {
            "dataset": dataset,
            "operator_sha256": operator_sha,
            "format_sha256": format_sha,
            "strategy_signature": iteration.get("strategy_signature"),
            "generated_ms": generated_ms,
            "source_run": str(run_dir),
        }
    )
    return {
        "schema_version": SCHEMA,
        "candidate_id": candidate_id,
        "dataset": dataset,
        "performance_contract": PERFORMANCE_CONTRACT,
        "protocol": protocol,
        "hardware": hardware,
        "kernel_provenance": "freshly_generated",
        "source": {
            "kind": source_kind,
            "run_id": run_dir.name,
            "run_dir": str(run_dir),
            "source_summary_path": str(source_summary_path),
            "source_summary_sha256": _sha256(source_summary_path),
            "iteration_dir": str(iteration_dir),
            "iteration_summary_path": str(iteration_summary_path),
            "iteration_summary_sha256": _sha256(iteration_summary_path),
        },
        "selection": {
            "format_iteration": format_iteration,
            "strategy_iteration": int(iteration.get("iteration") or 0),
            "strategy_signature": iteration.get("strategy_signature"),
        },
        "performance": {
            "generated_ms": generated_ms,
            "cusparse_ms": cusparse_ms,
            "speedup_vs_cusparse": speedup,
            "generated_ms_samples": timing.get("generated_ms_samples", []),
            "cusparse_ms_samples": timing.get("cusparse_ms_samples", []),
            "relative_spread_pct": timing.get("relative_spread_pct"),
        },
        "correctness": {
            "passed": True,
            "all_measurements_passed": correctness.get("all_measurements_passed"),
            "measurement_runs": correctness.get("measurement_runs"),
            "mismatches": correctness.get("mismatches"),
            "nonfinite": correctness.get("nonfinite"),
            "relative_l2_error": correctness.get("relative_l2_error"),
            "max_abs_normalized": correctness.get("max_abs_normalized"),
        },
        "matrix": {
            "path": result.get("matrix_path"),
            "shape": shape,
            "value_policy": result.get("value_policy"),
        },
        "format": {
            "format_name": format_spec.get("format_name"),
            "format_family": format_spec.get("format_family"),
            "atomic_type": atomic.get("type"),
            "atomic_layout": atomic.get("layout"),
            "params": format_spec.get("params", {}),
            "levels": format_spec.get("levels", []),
            "physical": physical_format,
            "backend_plan": backend,
            "kernel_visible_costs": packed,
        },
        "strategy": _redact_secrets(strategy),
        "artifacts": {
            "operator_path": str(operator_path),
            "operator_sha256": operator_sha,
            "operator_bytes": operator_path.stat().st_size,
            "format_path": str(format_path),
            "format_sha256": format_sha,
            "generation_manifest_path": str(generation_manifest_path),
            "generation_manifest_sha256": _sha256(generation_manifest_path),
        },
    }


def _extract_joint_candidates(
    run_dir: Path, datasets: set[str] | None
) -> list[dict[str, Any]]:
    candidates = []
    for summary_path in sorted((run_dir / "datasets").glob("*/joint_summary.json")):
        summary = _read_json(summary_path)
        dataset = str(summary.get("dataset") or summary_path.parent.name)
        if datasets is not None and dataset not in datasets:
            continue
        for record in summary.get("format_iterations", []):
            if not isinstance(record, dict):
                continue
            if record.get("best_correctness_passed") is not True:
                continue
            operator = record.get("best_operator_path")
            artifact = record.get("format_artifact")
            if not operator or not artifact:
                continue
            candidates.append(
                _candidate_from_iteration(
                    run_dir=run_dir,
                    dataset=dataset,
                    operator_path=_resolve_path(operator, run_dir),
                    format_path=_resolve_path(artifact, run_dir),
                    source_kind="joint_format_strategy",
                    source_summary_path=summary_path,
                    format_iteration=int(record.get("format_iteration")),
                    expected_generated_ms=float(record["best_generated_ms"]),
                    expected_speedup=float(record["best_speedup_vs_cusparse"]),
                )
            )
    return candidates


def _extract_historical_two_agent_candidates(
    run_dir: Path, datasets: set[str] | None
) -> list[dict[str, Any]]:
    comparison = run_dir / "sparsene_2agent_speedup_comparison.csv"
    if not comparison.is_file():
        return []
    candidates = []
    with comparison.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            dataset = str(row.get("dataset") or "")
            if datasets is not None and dataset not in datasets:
                continue
            if str(row.get("two_agent_correctness_passed", "")).lower() != "true":
                continue
            candidates.append(
                _candidate_from_iteration(
                    run_dir=run_dir,
                    dataset=dataset,
                    operator_path=_resolve_path(row["two_agent_operator_path"], run_dir),
                    source_kind="fixed_format_strategy",
                    source_summary_path=comparison,
                    expected_generated_ms=float(row["two_agent_generated_ms"]),
                    expected_cusparse_ms=float(row["two_agent_cusparse_ms"]),
                    expected_speedup=float(row["sparsene_2agent_speedup"]),
                )
            )
    return candidates


def extract_run_candidates(
    run_dir: Path, datasets: Iterable[str] | None = None
) -> list[dict[str, Any]]:
    """Extract all correctness-gated selected candidates from a supported run."""

    run_dir = run_dir.resolve()
    if not run_dir.is_dir():
        raise IncumbentRegistryError(f"missing source run: {run_dir}")
    selected = set(datasets) if datasets is not None else None
    joint = _extract_joint_candidates(run_dir, selected)
    if joint:
        return joint
    historical = _extract_historical_two_agent_candidates(run_dir, selected)
    if historical:
        return historical
    raise IncumbentRegistryError(
        f"unsupported run layout or no correct generated candidates: {run_dir}"
    )


def candidate_matches_scope(
    candidate: dict[str, Any],
    *,
    protocol: dict[str, Any] = DEFAULT_PROTOCOL,
    hardware_scope: dict[str, Any] = DEFAULT_HARDWARE_SCOPE,
) -> tuple[bool, list[str]]:
    issues = []
    if candidate.get("performance_contract") != PERFORMANCE_CONTRACT:
        issues.append("performance_contract")
    actual_protocol = candidate.get("protocol", {})
    for key, expected in protocol.items():
        if actual_protocol.get(key) != expected:
            issues.append(f"protocol.{key}")
    actual_hardware = candidate.get("hardware", {})
    for key in ("target", "compute_capability", "sm_count"):
        if actual_hardware.get(key) != hardware_scope.get(key):
            issues.append(f"hardware.{key}")
    correctness = candidate.get("correctness", {})
    if correctness.get("passed") is not True:
        issues.append("correctness.passed")
    if int(protocol.get("measurement_runs", 1)) > 1 and correctness.get(
        "all_measurements_passed"
    ) is not True:
        issues.append("correctness.all_measurements_passed")
    if candidate.get("kernel_provenance") != "freshly_generated":
        issues.append("kernel_provenance")
    return not issues, issues


def load_incumbent(registry_dir: Path, dataset: str) -> dict[str, Any] | None:
    path = registry_dir / "datasets" / dataset / "current.json"
    return _read_json(path) if path.is_file() else None


def compact_incumbent_feedback(record: dict[str, Any] | None) -> dict[str, Any]:
    """Return representation-level incumbent evidence without raw NCU details."""

    if not record:
        return {}
    return {
        "dataset": record.get("dataset"),
        "candidate_id": record.get("candidate_id"),
        "source": record.get("source"),
        "selection": record.get("selection"),
        "performance": record.get("performance"),
        "correctness": record.get("correctness"),
        "format": record.get("format"),
        "strategy": {
            key: record.get("strategy", {}).get(key)
            for key in (
                "tensor_placement",
                "pipeline",
                "compiler",
                "runtime_schedule",
            )
            if key in record.get("strategy", {})
        },
    }


def incumbent_as_joint_record(record: dict[str, Any]) -> dict[str, Any]:
    artifacts = record.get("registry_artifacts", {})
    return {
        "schema_version": record.get("schema_version"),
        "dataset": record.get("dataset"),
        "format_iteration": None,
        "status": "incumbent",
        "source_kind": "incumbent_registry",
        "source_run_id": record.get("source", {}).get("run_id"),
        "format_artifact": artifacts.get("format_path"),
        "format_artifact_sha256": record.get("artifacts", {}).get("format_sha256"),
        "format_signature": record.get("candidate_id"),
        "format": record.get("format", {}),
        "strategy_candidate_budget": 0,
        "strategy_candidates_evaluated": 0,
        "strategy_candidates_validated": 0,
        "strategy_candidates_incorrect": 0,
        "strategy_candidates_failed": 0,
        "best_strategy_iteration": record.get("selection", {}).get(
            "strategy_iteration"
        ),
        "best_generated_ms": record.get("performance", {}).get("generated_ms"),
        "best_speedup_vs_cusparse": record.get("performance", {}).get(
            "speedup_vs_cusparse"
        ),
        "best_operator_path": artifacts.get("operator_path"),
        "best_correctness_passed": record.get("correctness", {}).get("passed") is True,
        "best_result": {
            "strategy_iteration": record.get("selection", {}).get(
                "strategy_iteration"
            ),
            "timing": record.get("performance", {}),
            "correctness": record.get("correctness", {}),
            "physical_format": record.get("format", {}).get("physical", {}),
            "kernel_visible_format_costs": record.get("format", {}).get(
                "kernel_visible_costs", {}
            ),
            "effective_strategy": compact_incumbent_feedback(record).get(
                "strategy", {}
            ),
        },
    }


def _publish_version(registry_dir: Path, candidate: dict[str, Any]) -> dict[str, Any]:
    dataset = str(candidate["dataset"])
    candidate_id = str(candidate["candidate_id"])
    dataset_dir = registry_dir / "datasets" / dataset
    version_dir = dataset_dir / "versions" / candidate_id[:16]
    version_dir.mkdir(parents=True, exist_ok=True)
    source_artifacts = candidate["artifacts"]
    operator_source = _require_file(
        Path(source_artifacts["operator_path"]), "candidate operator"
    )
    format_source = _require_file(
        Path(source_artifacts["format_path"]), "candidate format"
    )
    manifest_source = _require_file(
        Path(source_artifacts["generation_manifest_path"]),
        "candidate generation manifest",
    )
    operator_target = version_dir / "operator.inc"
    format_target = version_dir / "format.format.json"
    strategy_target = version_dir / "strategy.json"
    manifest_target = version_dir / "generation_manifest.json"
    _atomic_copy(operator_source, operator_target)
    _atomic_copy(format_source, format_target)
    _atomic_write_json(strategy_target, candidate.get("strategy", {}))
    _atomic_write_json(manifest_target, _redact_secrets(_read_json(manifest_source)))
    if _sha256(operator_target) != source_artifacts["operator_sha256"]:
        raise IncumbentRegistryError(f"registry operator copy mismatch for {dataset}")
    if _sha256(format_target) != source_artifacts["format_sha256"]:
        raise IncumbentRegistryError(f"registry format copy mismatch for {dataset}")

    stored = dict(candidate)
    stored["registry_artifacts"] = {
        "version_dir": str(version_dir.resolve()),
        "operator_path": str(operator_target.resolve()),
        "format_path": str(format_target.resolve()),
        "strategy_path": str(strategy_target.resolve()),
        "generation_manifest_path": str(manifest_target.resolve()),
    }
    stored["registry_updated_at"] = _now()
    _atomic_write_json(version_dir / "record.json", stored)
    _atomic_copy(operator_target, dataset_dir / "best_operator.inc")
    _atomic_copy(format_target, dataset_dir / "best_format.format.json")
    _atomic_copy(strategy_target, dataset_dir / "best_strategy.json")
    _atomic_write_json(dataset_dir / "current.json", stored)
    return stored


def _registry_records(registry_dir: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted((registry_dir / "datasets").glob("*/current.json")):
        records.append(_read_json(path))
    return records


def _load_paper_sparsene_baseline(
    path: Path = DEFAULT_SPARSENE_PAPER_BASELINE,
) -> dict[str, float]:
    """Load the per-dataset Sparsene 1.0 bars published by the paper."""

    if not path.is_file():
        return {}
    values: dict[str, float] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            dataset = str(row.get("dataset") or "")
            if not dataset or dataset in values:
                raise IncumbentRegistryError(
                    f"invalid or duplicate paper baseline dataset: {dataset!r}"
                )
            if row.get("gpu") != "A100" or int(row.get("N") or 0) != 512:
                raise IncumbentRegistryError(
                    f"paper baseline scope mismatch for {dataset}"
                )
            value = float(row["paper_sparsene_speedup_vs_cusparse"])
            if value <= 0.0:
                raise IncumbentRegistryError(
                    f"non-positive paper baseline for {dataset}: {value}"
                )
            values[dataset] = value
    return values


def _summary_row(record: dict[str, Any]) -> dict[str, Any]:
    fmt = record.get("format", {})
    physical = fmt.get("physical", {}) if isinstance(fmt.get("physical"), dict) else {}
    strategy = record.get("strategy", {})
    placement = (
        strategy.get("tensor_placement", {})
        if isinstance(strategy.get("tensor_placement"), dict)
        else {}
    )
    runtime = (
        strategy.get("runtime_schedule", {})
        if isinstance(strategy.get("runtime_schedule"), dict)
        else {}
    )
    return {
        "dataset": record.get("dataset"),
        "generated_ms": record.get("performance", {}).get("generated_ms"),
        "cusparse_ms": record.get("performance", {}).get("cusparse_ms"),
        "speedup_vs_cusparse": record.get("performance", {}).get(
            "speedup_vs_cusparse"
        ),
        "relative_spread_pct": record.get("performance", {}).get(
            "relative_spread_pct"
        ),
        "source_run": record.get("source", {}).get("run_id"),
        "source_kind": record.get("source", {}).get("kind"),
        "format_iteration": record.get("selection", {}).get("format_iteration"),
        "strategy_iteration": record.get("selection", {}).get("strategy_iteration"),
        "format_name": fmt.get("format_name"),
        "format_family": fmt.get("format_family"),
        "atomic_type": fmt.get("atomic_type"),
        "logical_blk_m": physical.get("logical_blk_m"),
        "physical_blk_m": physical.get("physical_blk_m"),
        "blk_k": physical.get("blk_k"),
        "packing_mode": physical.get("packing_mode"),
        "row_reorder": physical.get("row_reorder"),
        "tile_b": placement.get("tile_b"),
        "execution_policy": runtime.get("execution_policy"),
        "operator_sha256": record.get("artifacts", {}).get("operator_sha256"),
        "format_sha256": record.get("artifacts", {}).get("format_sha256"),
        "operator_path": record.get("registry_artifacts", {}).get("operator_path"),
        "format_path": record.get("registry_artifacts", {}).get("format_path"),
    }


def _publish_registry_summary(
    registry_dir: Path,
    *,
    source_runs_considered: list[dict[str, Any]],
    last_update: dict[str, Any],
) -> None:
    records = _registry_records(registry_dir)
    rows = [_summary_row(record) for record in records]
    paper_sparsene = _load_paper_sparsene_baseline()
    for row in rows:
        row["sparsene1_paper_speedup_vs_cusparse"] = paper_sparsene.get(
            str(row["dataset"]), ""
        )
    fields = list(rows[0]) if rows else ["dataset"]
    csv_path = registry_dir / "summary.csv"
    temporary = csv_path.with_name(f".{csv_path.name}.tmp.{os.getpid()}")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, csv_path)

    speedups = [float(row["speedup_vs_cusparse"]) for row in rows]
    paper_speedups = [
        float(row["sparsene1_paper_speedup_vs_cusparse"])
        for row in rows
        if row["sparsene1_paper_speedup_vs_cusparse"] != ""
    ]
    geomean = (
        math.exp(sum(math.log(value) for value in speedups) / len(speedups))
        if speedups
        else None
    )
    registry = {
        "schema_version": SCHEMA,
        "performance_contract": PERFORMANCE_CONTRACT,
        "protocol": DEFAULT_PROTOCOL,
        "hardware_scope": DEFAULT_HARDWARE_SCOPE,
        "selection_rule": (
            "minimum positive median generated_ms among freshly generated candidates "
            "whose full outputs pass every independent correctness measurement"
        ),
        "updated_at": _now(),
        "counts": {
            "datasets": len(records),
            "faster_than_cusparse": sum(value > 1.0 for value in speedups),
        },
        "geometric_mean_speedup_vs_cusparse": geomean,
        "historical_sparsene1_baseline": {
            "scope": "paper A100, N=512, FP32, normalized to paper cuSPARSE",
            "source_path": str(DEFAULT_SPARSENE_PAPER_BASELINE.resolve()),
            "source_sha256": (
                _sha256(DEFAULT_SPARSENE_PAPER_BASELINE)
                if DEFAULT_SPARSENE_PAPER_BASELINE.is_file()
                else None
            ),
            "datasets": len(paper_speedups),
            "arithmetic_mean_speedup_vs_cusparse": (
                sum(paper_speedups) / len(paper_speedups)
                if paper_speedups
                else None
            ),
            "geometric_mean_speedup_vs_cusparse": (
                math.exp(
                    sum(math.log(value) for value in paper_speedups)
                    / len(paper_speedups)
                )
                if paper_speedups
                else None
            ),
            "paper_reported_arithmetic_mean": 2.19,
        },
        "source_runs_considered": source_runs_considered,
        "last_update": last_update,
        "datasets": {
            str(record["dataset"]): {
                "candidate_id": record.get("candidate_id"),
                "generated_ms": record.get("performance", {}).get("generated_ms"),
                "speedup_vs_cusparse": record.get("performance", {}).get(
                    "speedup_vs_cusparse"
                ),
                "source_run": record.get("source", {}).get("run_id"),
                "current_record": str(
                    (registry_dir / "datasets" / str(record["dataset"]) / "current.json").resolve()
                ),
            }
            for record in records
        },
    }
    _atomic_write_json(registry_dir / "registry.json", registry)
    _atomic_write_json(registry_dir / "last_update.json", last_update)

    lines = [
        "# SparseKG N=512 FP32 cross-run incumbents",
        "",
        f"- Correctness-gated datasets: **{len(rows)}**",
        f"- GeoMean over cuSPARSE: **{geomean:.4f}×**" if geomean else "- GeoMean: n/a",
        f"- Faster than cuSPARSE: **{sum(value > 1.0 for value in speedups)}/{len(rows)}**",
        (
            "- Sparsene 1.0 paper baseline: "
            f"arithmetic mean **{sum(paper_speedups) / len(paper_speedups):.4f}×** "
            "(reported as 2.19×); "
            f"geometric mean **{math.exp(sum(math.log(value) for value in paper_speedups) / len(paper_speedups)):.4f}×**."
            if paper_speedups
            else "- Sparsene 1.0 paper baseline: unavailable."
        ),
        "- Ranking: minimum warmed median generated-kernel time; lower `generated_ms` is better.",
        "- Sparsene 1.0 is the published A100/N=512/FP32 speedup over its paper cuSPARSE baseline; it is historical context, not a timing from the incumbent run.",
        "",
        "| Dataset | Kernel ms | cuSPARSE ms | SparseKG speedup | Sparsene 1.0 paper speedup | Format | Physical layout | Strategy | Source run |",
        "|---|---:|---:|---:|---:|---|---|---|---|",
    ]
    for row in rows:
        layout = (
            f"{row['logical_blk_m']}x{row['blk_k']} "
            f"{row['packing_mode']} reorder={row['row_reorder']}"
        )
        strategy_text = (
            f"iter={row['strategy_iteration']}, tile_b={row['tile_b']}, "
            f"policy={row['execution_policy']}"
        )
        lines.append(
            f"| `{row['dataset']}` | {float(row['generated_ms']):.6g} | "
            f"{float(row['cusparse_ms']):.6g} | {float(row['speedup_vs_cusparse']):.4f}× | "
            + (
                f"{float(row['sparsene1_paper_speedup_vs_cusparse']):.4f}× | "
                if row["sparsene1_paper_speedup_vs_cusparse"] != ""
                else "n/a | "
            )
            + f"`{row['format_name']}` ({row['atomic_type']}) | {layout} | "
            f"{strategy_text} | `{row['source_run']}` |"
        )
    _atomic_write_text(registry_dir / "summary.md", "\n".join(lines) + "\n")

    readme = """# Cross-run incumbent registry

This directory is a maintained artifact store, not an individual experiment.
Only freshly generated kernels using `sparsene.kernel_only_post_conversion.v1`
with N=512 FP32/TF32, warmup=3, repeat=32, three median-ranked measurements,
SM80 compute capability 8.0, 108 SMs, and full-output correctness in every
measurement are eligible. A candidate replaces the current dataset incumbent
only when its `generated_ms` is strictly lower.

Authoritative files:

- `registry.json`: registry contract, sources, counts, and current pointers;
- `summary.csv` / `summary.md`: human- and script-readable leaderboard;
- `incumbent_three_method_speedup_comparison.pdf` / `.png`: paper-style
  cuSPARSE/Sparsene/SparseKG comparison derived from the leaderboard;
- `incumbent_three_method_speedup_comparison_data.csv` / `_audit.json`:
  exact plotting data and source hashes;
- `incumbent_three_method_speedup_comparison_table.tex`: paper-ready table with
  cuSPARSE stated as the normalized 1.0 baseline, only Sparsene/SparseKG shown,
  and the best full-precision displayed speedup in each row typeset in bold;
- `datasets/<dataset>/current.json`: authoritative full current record;
- `datasets/<dataset>/best_operator.inc`: convenient current operator copy;
- `datasets/<dataset>/best_format.format.json`: convenient current Format DSL;
- `datasets/<dataset>/best_strategy.json`: convenient current strategy;
- `datasets/<dataset>/versions/`: immutable historical incumbent snapshots;
- `history.jsonl`: insertion/replacement audit trail.

Update from one or more completed runs:

```bash
source /mnt/lustre-client/huoda/venv/llm-sog/bin/activate
PYTHONPATH=python python3 python/test/update_e2e_incumbents.py \\
  --source-run logs/e2e_runs/<run_id>
```

Rebuild the paper-style comparison after the leaderboard changes:

```bash
PYTHONPATH=python python3 python/test/build_incumbent_three_method_comparison.py
```

The joint Format/Strategy driver reads and updates this registry by default.
Research-only formats remain preserved in their run directories, but cannot
replace an executable incumbent until a generated kernel passes the complete
protocol above.  The Sparsene 1.0 column is historical A100/N=512/FP32 paper
context extracted from the published vector figure and normalized to the
paper's own cuSPARSE baseline; it is not re-ranked as a registry candidate.
"""
    _atomic_write_text(registry_dir / "README.md", readme)


def update_registry(
    registry_dir: Path,
    candidates: Iterable[dict[str, Any]],
    *,
    min_improvement_pct: float = 0.0,
) -> dict[str, Any]:
    """Apply candidates under an exclusive lock and publish refreshed summaries."""

    if min_improvement_pct < 0.0:
        raise IncumbentRegistryError("min_improvement_pct must be nonnegative")
    registry_dir = registry_dir.resolve()
    registry_dir.mkdir(parents=True, exist_ok=True)
    lock_path = registry_dir / ".update.lock"
    report: dict[str, Any] = {
        "schema_version": SCHEMA,
        "time": _now(),
        "candidates_considered": 0,
        "inserted": [],
        "replaced": [],
        "retained": [],
        "rejected": [],
    }
    candidate_list = list(candidates)
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        previous_registry = (
            _read_json(registry_dir / "registry.json")
            if (registry_dir / "registry.json").is_file()
            else {}
        )
        source_runs = list(previous_registry.get("source_runs_considered", []))
        source_index = {item.get("run_dir"): item for item in source_runs}
        for candidate in candidate_list:
            report["candidates_considered"] += 1
            compatible, issues = candidate_matches_scope(candidate)
            dataset = str(candidate.get("dataset"))
            if not compatible:
                report["rejected"].append({"dataset": dataset, "issues": issues})
                continue
            source = candidate.get("source", {})
            source_index[source.get("run_dir")] = {
                "run_id": source.get("run_id"),
                "run_dir": source.get("run_dir"),
                "source_kind": source.get("kind"),
                "source_summary_path": source.get("source_summary_path"),
                "source_summary_sha256": source.get("source_summary_sha256"),
            }
            current = load_incumbent(registry_dir, dataset)
            new_ms = float(candidate["performance"]["generated_ms"])
            old_ms = (
                float(current["performance"]["generated_ms"])
                if current is not None
                else None
            )
            threshold = (
                old_ms * (1.0 - min_improvement_pct / 100.0)
                if old_ms is not None
                else None
            )
            if old_ms is not None and not new_ms < float(threshold):
                report["retained"].append(
                    {
                        "dataset": dataset,
                        "incumbent_ms": old_ms,
                        "candidate_ms": new_ms,
                        "candidate_source_run": source.get("run_id"),
                    }
                )
                continue
            stored = _publish_version(registry_dir, candidate)
            event = {
                "time": _now(),
                "action": "inserted" if current is None else "replaced",
                "dataset": dataset,
                "candidate_id": stored["candidate_id"],
                "source_run": stored.get("source", {}).get("run_id"),
                "new_generated_ms": new_ms,
                "old_candidate_id": current.get("candidate_id") if current else None,
                "old_source_run": current.get("source", {}).get("run_id") if current else None,
                "old_generated_ms": old_ms,
                "speedup_over_previous": old_ms / new_ms if old_ms else None,
            }
            report[event["action"]].append(event)
            with (registry_dir / "history.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, ensure_ascii=False) + "\n")
        _publish_registry_summary(
            registry_dir,
            source_runs_considered=sorted(
                source_index.values(), key=lambda item: str(item.get("run_id"))
            ),
            last_update=report,
        )
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    return report


def update_registry_from_runs(
    registry_dir: Path,
    source_runs: Iterable[Path],
    *,
    datasets: Iterable[str] | None = None,
    min_improvement_pct: float = 0.0,
) -> dict[str, Any]:
    """Extract and apply source runs in order so replacement history is retained."""

    combined = {
        "schema_version": SCHEMA,
        "time": _now(),
        "source_reports": [],
        "inserted": 0,
        "replaced": 0,
        "retained": 0,
        "rejected": 0,
    }
    for run_dir in source_runs:
        candidates = extract_run_candidates(Path(run_dir), datasets=datasets)
        report = update_registry(
            registry_dir,
            candidates,
            min_improvement_pct=min_improvement_pct,
        )
        combined["source_reports"].append(
            {
                "run_dir": str(Path(run_dir).resolve()),
                "candidates_extracted": len(candidates),
                "report": report,
            }
        )
        for key in ("inserted", "replaced", "retained", "rejected"):
            combined[key] += len(report[key])
    return combined


def refresh_registry_summary(registry_dir: Path) -> dict[str, Any]:
    """Republish derived tables without changing candidates or update history."""

    registry_dir = registry_dir.resolve()
    registry_path = registry_dir / "registry.json"
    if not registry_path.is_file():
        raise IncumbentRegistryError(f"missing registry: {registry_path}")
    lock_path = registry_dir / ".update.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        previous = _read_json(registry_path)
        _publish_registry_summary(
            registry_dir,
            source_runs_considered=list(previous.get("source_runs_considered", [])),
            last_update=dict(previous.get("last_update", {})),
        )
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    return _read_json(registry_path)
