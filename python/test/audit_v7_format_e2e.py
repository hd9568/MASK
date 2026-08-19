#!/usr/bin/env python3
"""Audit the provenance and numerical gate of a saved v7 format E2E run.

This is intentionally a read-only audit: it does not regenerate an operator,
recompile, or launch the GPU.  It verifies that each recorded validated row
still forms an unbroken chain from its run-owned format artifact to the exact
generated include, CMake cache, executable, result JSON, and full-output
correctness decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from sparsene.formats.custom.loader import load_run_format


SCHEMA = "sparsene.v7_format_e2e_provenance_audit.v1"
REQUIRED_SETTINGS = {
    "n": 512,
    "warmup": 3,
    "repeat": 32,
    "jobs": 1,
    "runtime_max_tiles_per_segment": 0,
    "execution_policy": "auto",
    "task_group_tiles": 64,
}


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _same_path(left: str | Path, right: str | Path) -> bool:
    return Path(left).resolve() == Path(right).resolve()


def _check(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def _audit_validated_row(row: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    dataset = row["dataset"]
    errors: list[str] = []
    artifact = run_dir / "formats" / f"{dataset}.format.json"
    generation_path = Path(row["generation_manifest"])
    result_path = Path(row["e2e_result"])
    _check(artifact.is_file(), "missing run-owned format artifact", errors)
    _check(generation_path.is_file(), "missing generation manifest", errors)
    _check(result_path.is_file(), "missing E2E result JSON", errors)
    if errors:
        return {"dataset": dataset, "status": "fail", "errors": errors}

    generation = _json(generation_path)
    result = _json(result_path)
    try:
        _spec, _format_obj, loaded_artifact, artifact_sha = load_run_format(run_dir, dataset)
        _check(_same_path(loaded_artifact, artifact), "loader resolved another format artifact", errors)
    except Exception as exc:
        errors.append(f"current format artifact fails DSL/load validation: {exc}")
        artifact_sha = _sha256(artifact)
    source = generation.get("format_source", {})
    _check(row.get("artifact_sha256") == artifact_sha, "status artifact SHA differs from current artifact", errors)
    _check(source.get("artifact_sha256") == artifact_sha, "generation artifact SHA differs from current artifact", errors)
    _check(_same_path(source.get("artifact_path", ""), artifact), "generation manifest points to another format artifact", errors)

    operator = Path(generation.get("operator_path", ""))
    _check(operator.is_file(), "generated operator include is missing", errors)
    if operator.is_file():
        _check(generation.get("operator_sha256") == _sha256(operator), "generated operator SHA differs from manifest", errors)
    _check(_same_path(result.get("operator_path", ""), operator), "E2E result points to another operator include", errors)

    plan = generation.get("backend_plan")
    _check(plan == result.get("backend_plan"), "generation and E2E backend plans differ", errors)
    _check(row.get("backend") == plan.get("backend"), "status backend differs from generation plan", errors)
    _check(row.get("codegen_format_name") == plan.get("codegen_format_name"), "status codegen format differs from generation plan", errors)
    _check(row.get("packing_mode") == plan.get("packing_mode"), "status packing mode differs from generation plan", errors)
    _check(row.get("logical_blk_m") == plan.get("logical_blk_m"), "status BLK_M differs from generation plan", errors)
    _check(row.get("logical_blk_k") == plan.get("logical_blk_k"), "status BLK_K differs from generation plan", errors)

    work_dir = generation_path.parent
    build_cache = work_dir / "build" / "CMakeCache.txt"
    executable = Path(result.get("executable", ""))
    _check(build_cache.is_file(), "CMake cache is missing", errors)
    _check(executable.is_file(), "compiled custom-format-e2e executable is missing", errors)
    if build_cache.is_file():
        cache = build_cache.read_text(encoding="utf-8", errors="replace")
        _check(str(operator) in cache, "CMake cache does not name the generated operator", errors)
    for name, needle in (("compile.log", "Built target custom-format-e2e"), ("run.log", "correctness=PASS")):
        log = work_dir / name
        _check(log.is_file(), f"missing {name}", errors)
        if log.is_file():
            _check(needle in log.read_text(encoding="utf-8", errors="replace"), f"{name} lacks {needle!r}", errors)

    timing = result.get("timing", {})
    runtime = result.get("runtime_execution", {})
    correctness = result.get("correctness", {})
    _check(result.get("shape", {}).get("N") == REQUIRED_SETTINGS["n"], "N is not 512", errors)
    _check(timing.get("contract") == "sparsene.kernel_only_post_conversion.v1", "wrong timing contract", errors)
    _check(timing.get("warmup") == REQUIRED_SETTINGS["warmup"], "warmup is not 3", errors)
    _check(timing.get("repeat") == REQUIRED_SETTINGS["repeat"], "repeat is not 32", errors)
    _check(runtime == {"max_tiles_per_segment": 0, "execution_policy": "auto", "task_group_tiles": 64}, "runtime execution settings differ", errors)
    _check(correctness.get("passed") is True, "correctness gate did not pass", errors)
    _check(correctness.get("nonfinite") == 0, "output contains non-finite values", errors)
    _check(correctness.get("elements", 0) > 0, "no output elements were compared", errors)
    _check(correctness.get("relative_l2_error", float("inf")) <= correctness.get("tf32_max_relative_l2", -1), "relative-L2 exceeds recorded TF32 bound", errors)
    _check(correctness.get("max_abs_normalized", float("inf")) <= correctness.get("tf32_max_abs_normalized", -1), "normalized max error exceeds recorded TF32 bound", errors)
    _check(row.get("correctness_passed") is True, "status does not record correctness pass", errors)
    _check(row.get("mismatches") == correctness.get("mismatches"), "status mismatch count differs from result", errors)
    _check(row.get("relative_l2_error") == correctness.get("relative_l2_error"), "status relative-L2 differs from result", errors)

    return {
        "dataset": dataset,
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "format_name": row.get("format_name"),
        "backend": plan.get("backend"),
        "elements_checked": correctness.get("elements"),
        "strict_mismatches": correctness.get("mismatches"),
        "relative_l2_error": correctness.get("relative_l2_error"),
        "max_abs_normalized": correctness.get("max_abs_normalized"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    status_path = run_dir / "operator_e2e_v7_all_status.json"
    status = _json(status_path)
    settings = status.get("settings", {})
    setting_errors = [
        f"settings.{key}={settings.get(key)!r}, expected {expected!r}"
        for key, expected in REQUIRED_SETTINGS.items()
        if settings.get(key) != expected
    ]
    manifest = _json(run_dir / "run_manifest.json")
    rows = status.get("results", [])
    seen = [row.get("dataset") for row in rows]
    manifest_datasets = manifest.get("datasets", [])
    coverage_errors: list[str] = []
    _check(seen == manifest_datasets, "status rows do not exactly match run-manifest order", coverage_errors)
    audited = [_audit_validated_row(row, run_dir) for row in rows if row.get("status") == "validated"]
    unsupported = [
        {"dataset": row.get("dataset"), "error": row.get("error")}
        for row in rows if row.get("status") == "unsupported"
    ]
    failures = [item for item in audited if item["status"] != "pass"]
    total_elements = sum(int(item.get("elements_checked", 0)) for item in audited)
    total_strict_mismatches = sum(int(item.get("strict_mismatches", 0)) for item in audited)
    payload = {
        "schema_version": SCHEMA,
        "run_dir": str(run_dir),
        "read_only": True,
        "performance_contract": status.get("performance_contract"),
        "required_settings": REQUIRED_SETTINGS,
        "recorded_settings": settings,
        "setting_errors": setting_errors,
        "coverage_errors": coverage_errors,
        "summary": {
            "requested_formats": len(manifest_datasets),
            "validated_and_audited": len(audited),
            "audit_passed": len(audited) - len(failures),
            "audit_failed": len(failures),
            "backend_unsupported": len(unsupported),
            "total_output_elements_compared": total_elements,
            "total_strict_pointwise_mismatches": total_strict_mismatches,
        },
        "unsupported": unsupported,
        "results": audited,
    }
    output = run_dir / "operator_e2e_v7_provenance_audit.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False))
    if setting_errors or coverage_errors or failures:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
