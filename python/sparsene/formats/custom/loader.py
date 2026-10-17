"""Load an exported custom-format module without invoking an LLM."""

from __future__ import annotations

import hashlib
from importlib import import_module
import json
from pathlib import Path
import re
from types import ModuleType
from typing import Any

from sparsene.format.format import Format
from sparsene.formats.dsl import build_format_from_spec, validate_format_spec


def module_stem(dataset: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_]", "_", dataset)
    if not stem or stem[0].isdigit():
        stem = "matrix_" + stem
    return stem


def load_custom_format(dataset: str) -> tuple[dict[str, Any], Format, ModuleType]:
    module = import_module(f"sparsene.formats.custom.{module_stem(dataset)}")
    spec = module.FORMAT_SPEC
    format_obj = module.FORMAT
    if not isinstance(spec, dict) or not isinstance(format_obj, Format):
        raise TypeError(f"Malformed custom format module for {dataset!r}")
    if spec.get("dataset") != dataset:
        raise ValueError(
            f"Custom module dataset mismatch: requested {dataset!r}, "
            f"found {spec.get('dataset')!r}"
        )
    return spec, format_obj, module


def load_run_format(
    run_dir: Path, dataset: str
) -> tuple[dict[str, Any], Format, Path, str]:
    """Load the immutable format artifact owned by one audited run.

    ``formats/custom`` is a convenience export location and can only hold one
    run's modules at a time.  E2E generation must instead consume the exact
    JSON artifact beneath the requested run so a later export cannot silently
    substitute a different Agent decision.
    """
    artifact_path = run_dir / "formats" / f"{dataset}.format.json"
    if not artifact_path.is_file():
        raise FileNotFoundError(f"Missing format artifact for {dataset}: {artifact_path}")
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Format artifact must be an object: {artifact_path}")
    if payload.get("dataset") != dataset:
        raise ValueError(
            f"Format artifact dataset mismatch: requested {dataset!r}, "
            f"found {payload.get('dataset')!r}"
        )
    validation = validate_format_spec(payload)
    if validation["status"] != "valid":
        raise ValueError(
            f"Invalid run-owned Format DSL for {dataset}: "
            + "; ".join(validation["errors"])
        )
    return (
        payload,
        build_format_from_spec(payload),
        artifact_path.resolve(),
        hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
    )


def custom_dir() -> Path:
    return Path(__file__).resolve().parent
