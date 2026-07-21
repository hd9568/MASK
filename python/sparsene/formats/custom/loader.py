"""Load an exported custom-format module without invoking an LLM."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
import re
from types import ModuleType
from typing import Any

from sparsene.format.format import Format


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


def custom_dir() -> Path:
    return Path(__file__).resolve().parent
