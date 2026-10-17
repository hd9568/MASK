"""Public construction helpers for the JSON Format DSL.

The format-design agent, checked-in custom formats, and code-generation tools
must all construct the same :class:`Format` tree.  This module deliberately has
no agent/API dependency so a saved design remains executable offline.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from sympy import Expr, Integer, Symbol, sympify

from sparsene.format.format import (
    AtomicFormat,
    Axis,
    Direction,
    Format,
    atomic_format,
    df_axis,
    dv_axis,
    sf_axis,
    sv_axis,
)


AXIS_TYPES = ("df", "dv", "sf", "sv")
ATOMIC_TYPES = ("coo", "mco", "csr", "ell", "dia", "dense", "custom")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_FORMAT_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*$")
_LENGTH_EXPR_RE = re.compile(r"^[A-Za-z0-9_+\-*/() ]+$")
_REORDER_RE = re.compile(r"^[A-Za-z0-9_.:-]+$")


def parse_length_expr(length: Any, allowed_symbols: Sequence[str]) -> Expr:
    """Parse the intentionally small arithmetic language used by axis lengths."""
    text = str(length).strip()
    if not text:
        raise ValueError("length expression is empty")
    if not _LENGTH_EXPR_RE.fullmatch(text) or "**" in text or "//" in text:
        raise ValueError(f"unsafe or unsupported length expression: {text!r}")
    names = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text))
    unknown = sorted(names - set(allowed_symbols))
    if unknown:
        raise ValueError(f"undeclared symbols in length expression {text!r}: {unknown}")
    expr = sympify(
        text,
        locals={name: Symbol(name) for name in allowed_symbols},
        evaluate=False,
    )
    if isinstance(expr, Expr):
        return expr
    if isinstance(expr, int):
        return Integer(expr)
    raise ValueError(f"length expression is not symbolic/integer: {text!r}")


def validate_format_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Validate DSL syntax and cross references without changing the proposal."""
    errors: list[str] = []
    warnings: list[str] = []
    format_name = spec.get("format_name")
    levels = spec.get("levels")
    atomic = spec.get("atomic")
    params = spec.get("params")

    if not isinstance(format_name, str) or not _FORMAT_NAME_RE.fullmatch(format_name):
        errors.append("format_name must be a valid identifier-like name")
    if not isinstance(params, dict):
        errors.append("params must be an object")
        params = {}
    param_names: list[str] = []
    for name, value in params.items():
        if not isinstance(name, str) or not _IDENTIFIER_RE.fullmatch(name):
            errors.append(f"invalid parameter name: {name!r}")
            continue
        param_names.append(name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            errors.append(f"parameter {name} must be a concrete positive JSON integer")

    allowed_symbols = ["M", "K", "N", *param_names]
    if not isinstance(levels, list) or not 1 <= len(levels) <= 6:
        errors.append("levels must contain between one and six Format levels")
        levels = []

    deepest_axes: dict[str, Mapping[str, Any]] = {}
    for index, level in enumerate(levels, start=1):
        if not isinstance(level, dict):
            errors.append(f"level {index} must be an object")
            continue
        if level.get("level") != index:
            errors.append(f"level numbering must be sequential; expected {index}")
        if not isinstance(level.get("name"), str) or not _IDENTIFIER_RE.fullmatch(level["name"]):
            errors.append(f"level {index} has an invalid name")
        axes = level.get("axes")
        if not isinstance(axes, list) or not 2 <= len(axes) <= 8:
            errors.append(f"level {index} must contain between two and eight axes")
            continue
        names_seen: set[str] = set()
        directions: set[str] = set()
        level_axes: dict[str, Mapping[str, Any]] = {}
        for axis_index, axis in enumerate(axes, start=1):
            path = f"level {index} axis {axis_index}"
            if not isinstance(axis, dict):
                errors.append(f"{path} must be an object")
                continue
            axis_name = axis.get("name")
            if not isinstance(axis_name, str) or not _IDENTIFIER_RE.fullmatch(axis_name):
                errors.append(f"{path} has an invalid name")
                continue
            if axis_name in names_seen:
                errors.append(f"level {index} repeats axis name {axis_name}")
            names_seen.add(axis_name)
            if axis.get("type") not in AXIS_TYPES:
                errors.append(f"{path} has unsupported type {axis.get('type')!r}")
            direction = axis.get("direction")
            if direction not in ("ROW", "COL"):
                errors.append(f"{path} has unsupported direction {direction!r}")
            else:
                directions.add(direction)
            try:
                parse_length_expr(axis.get("length"), allowed_symbols)
            except Exception as exc:
                errors.append(f"{path}: {exc}")
            reorder = axis.get("reorder")
            if reorder is not None and (
                not isinstance(reorder, str) or not _REORDER_RE.fullmatch(reorder)
            ):
                errors.append(f"{path} has an invalid reorder identifier")
            level_axes[axis_name] = axis
        if directions != {"ROW", "COL"}:
            errors.append(f"level {index} must represent both ROW and COL directions")
        if index == len(levels):
            deepest_axes = level_axes

    if not isinstance(atomic, dict):
        errors.append("atomic must be an object")
        atomic = {}
    atomic_type = atomic.get("type")
    if atomic_type not in ATOMIC_TYPES:
        errors.append(f"unsupported atomic type: {atomic_type!r}")
    if atomic.get("layout", "row_major") not in ("row_major", "col_major"):
        errors.append("atomic.layout must be row_major or col_major")
    if atomic_type == "custom":
        custom_spec = atomic.get("custom_spec")
        if not isinstance(custom_spec, dict):
            errors.append("custom atomic requires a custom_spec object")
        else:
            for field_name in ("value_arrays", "metadata_arrays"):
                value = custom_spec.get(field_name)
                if not isinstance(value, list) or not value:
                    errors.append(f"custom atomic requires non-empty custom_spec.{field_name}")
            for field_name in ("logical_mapping", "restore_algorithm"):
                if not str(custom_spec.get(field_name, "")).strip():
                    errors.append(f"custom atomic requires custom_spec.{field_name}")
    x_axis = atomic.get("x_axis")
    y_axis = atomic.get("y_axis")
    if x_axis not in deepest_axes:
        errors.append("atomic.x_axis must reference an axis in the deepest level")
    elif deepest_axes[x_axis].get("direction") != "ROW":
        errors.append("atomic.x_axis must reference a ROW axis")
    if y_axis not in deepest_axes:
        errors.append("atomic.y_axis must reference an axis in the deepest level")
    elif deepest_axes[y_axis].get("direction") != "COL":
        errors.append("atomic.y_axis must reference a COL axis")
    if x_axis == y_axis and x_axis is not None:
        errors.append("atomic.x_axis and atomic.y_axis must be different")

    return {
        "status": "valid" if not errors else "invalid",
        "errors": errors,
        "warnings": warnings,
        "level_count": len(levels),
        "axis_count": sum(
            len(level.get("axes", []))
            for level in levels
            if isinstance(level, dict) and isinstance(level.get("axes"), list)
        ),
        "atomic_type": atomic_type,
    }


def _build_axis(axis: Mapping[str, Any], allowed_symbols: Sequence[str]) -> Axis:
    direction = Direction.ROW if axis["direction"] == "ROW" else Direction.COL
    constructors = {"df": df_axis, "dv": dv_axis, "sf": sf_axis, "sv": sv_axis}
    return constructors[str(axis["type"])](
        str(axis["name"]),
        direction,
        parse_length_expr(axis["length"], allowed_symbols),
        axis.get("reorder"),
    )


def _build_atomic(atomic: Mapping[str, Any]) -> AtomicFormat:
    if atomic["type"] == "custom":
        col_major = atomic.get("layout", "row_major") == "col_major"
        return AtomicFormat(
            name=str(atomic.get("name", "custom")),
            atomic_type="custom",
            axes=[
                Axis(
                    name=str(atomic["x_axis"]),
                    direction=Direction.COL if col_major else Direction.ROW,
                    is_sparse=False,
                    is_varlen=False,
                ),
                Axis(
                    name=str(atomic["y_axis"]),
                    direction=Direction.ROW if col_major else Direction.COL,
                    is_sparse=False,
                    is_varlen=False,
                ),
            ],
        )
    return atomic_format(
        str(atomic["type"]),
        str(atomic["x_axis"]),
        str(atomic["y_axis"]),
        col_major=atomic.get("layout", "row_major") == "col_major",
    )


def build_format_from_spec(spec: Mapping[str, Any]) -> Format:
    """Build a frontend Format from a validated JSON-compatible mapping."""
    validation = validate_format_spec(spec)
    if validation["status"] != "valid":
        raise ValueError("Invalid Format DSL: " + "; ".join(validation["errors"]))
    params = spec["params"]
    allowed_symbols = ["M", "K", "N", *params.keys()]
    current: Format | AtomicFormat = _build_atomic(spec["atomic"])
    for level in reversed(spec["levels"]):
        current = Format(
            name=level["name"],
            axes=[_build_axis(axis, allowed_symbols) for axis in level["axes"]],
            child=current,
        )
    if not isinstance(current, Format):  # pragma: no cover
        raise ValueError("Top-level design must be a Format")
    return current
