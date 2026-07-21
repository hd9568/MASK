"""Semantic Format-DSL to executable-backend selection.

Backend matching is based on hierarchy/atomic semantics, never on a dataset or
an LLM-provided format name.  A backend plan also makes explicit any legal
physical padding needed by the target hardware.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class BackendPlan:
    backend: str
    codegen_format_name: str
    target: str
    compute_capability: int
    logical_blk_m: int
    logical_blk_k: int
    physical_blk_m: int
    physical_blk_k: int
    tile_n: int
    mma_m: int
    mma_n: int
    mma_k: int
    max_tiles_per_execution_segment: int
    packing_mode: str
    row_reorder: str | None
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _axis_map(level: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(axis.get("name")): axis for axis in level.get("axes", [])}


def _expect_axis(
    axes: Mapping[str, Mapping[str, Any]],
    name: str,
    axis_type: str,
    direction: str,
) -> Mapping[str, Any]:
    axis = axes.get(name)
    if axis is None:
        raise ValueError(f"TCF-MCO backend requires axis {name}")
    if axis.get("type") != axis_type or axis.get("direction") != direction:
        raise ValueError(
            f"TCF-MCO axis {name} must be {axis_type}/{direction}, got "
            f"{axis.get('type')}/{axis.get('direction')}"
        )
    return axis


def _expect_axis_type_in(
    axes: Mapping[str, Mapping[str, Any]],
    name: str,
    axis_types: set[str],
    direction: str,
) -> Mapping[str, Any]:
    axis = axes.get(name)
    if axis is None:
        raise ValueError(f"MCO tensor-core backend requires axis {name}")
    if axis.get("type") not in axis_types or axis.get("direction") != direction:
        raise ValueError(
            f"MCO tensor-core axis {name} must be one of {sorted(axis_types)}/{direction}, "
            f"got {axis.get('type')}/{axis.get('direction')}"
        )
    return axis


def select_backend(
    spec: Mapping[str, Any],
    *,
    target: str = "sm80",
    tile_n: int = 64,
    max_tiles_per_execution_segment: int = 32,
) -> BackendPlan:
    """Select an executable backend for a validated Format DSL mapping."""
    levels = spec.get("levels", [])
    atomic = spec.get("atomic", {})
    params = spec.get("params", {})
    atomic_type = atomic.get("type")
    if atomic_type not in {"mco", "dense"}:
        raise ValueError(
            "No executable backend for this design yet: current generalized route requires "
            "an MCO or dense atomic leaf"
        )
    x_outer: Mapping[str, Any]
    packing_mode: str
    backend: str
    codegen_format_name: str
    if len(levels) == 2:
        outer = _axis_map(levels[0])
        inner = _axis_map(levels[1])
        x_outer = _expect_axis(outer, "X_o", "df", "ROW")
        _expect_axis(outer, "Y_s", "sv", "COL")
        _expect_axis(outer, "X_i", "df", "ROW")
        # A fixed logical upper bound (df) and an explicitly variable block
        # sequence (dv) lower to the same physical per-window offset ABI.
        _expect_axis_type_in(inner, "Y_so", {"df", "dv"}, "COL")
        _expect_axis(inner, "Y_si", "df", "COL")
        _expect_axis(inner, "X_i", "df", "ROW")
        if atomic.get("x_axis") != "X_i" or atomic.get("y_axis") != "Y_si":
            raise ValueError("TCF leaf must map x_axis=X_i and y_axis=Y_si")
        if atomic_type == "dense":
            packing_mode = "column_condensed_dense"
            backend = "sr_bcrs_dense_tensorcore_v1"
            codegen_format_name = "SR_BCRS"
        else:
            packing_mode = "column_condensed"
            backend = "tcf_mco_tensorcore_v1"
            codegen_format_name = "BIT_TCF"
    elif len(levels) == 1:
        if atomic_type != "mco":
            raise ValueError("The one-level active-block backend requires an MCO atomic leaf")
        block = _axis_map(levels[0])
        x_outer = _expect_axis(block, "X_o", "df", "ROW")
        _expect_axis(block, "Y_o_s", "sv", "COL")
        _expect_axis(block, "X_i", "df", "ROW")
        _expect_axis(block, "Y_i", "df", "COL")
        if atomic.get("x_axis") != "X_i" or atomic.get("y_axis") != "Y_i":
            raise ValueError("BIT-BSR MCO leaf must map x_axis=X_i and y_axis=Y_i")
        packing_mode = "active_fixed_blocks"
        backend = "bit_bsr_mco_tensorcore_v1"
        codegen_format_name = "BIT_BSR"
    else:
        raise ValueError(
            "No executable MCO backend for this hierarchy: expected two-level "
            "column condensation or one-level active fixed blocks"
        )
    if atomic.get("layout", "row_major") != "row_major":
        raise ValueError("SM80 tensor-core backends require a row-major atomic leaf")

    row_reorders = {
        axis.get("reorder")
        for level in levels
        for axis in level.get("axes", [])
        if axis.get("reorder") is not None
    }
    supported_reorders = {"row_rcm", "first_col_sort", "jaccard_greedy"}
    if len(row_reorders) > 1 or not row_reorders.issubset(supported_reorders):
        raise ValueError(f"Unsupported reorder contract for MCO backend: {sorted(row_reorders)}")
    row_reorder = next(iter(row_reorders), None)
    if row_reorder and x_outer.get("reorder") != row_reorder:
        raise ValueError("row reorder must be attached to the outer X_o row axis")

    try:
        logical_m = int(params["BLK_M"])
        logical_k = int(params["BLK_K"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Tensor-core backend requires concrete integer BLK_M and BLK_K") from exc
    if target not in {"sm80", "a800"}:
        raise ValueError(f"Unknown target {target!r}; available target: sm80/a800")
    if logical_m not in {8, 16, 32}:
        raise ValueError(
            f"SM80 tensor-core backend supports logical BLK_M 8, 16, or 32, got {logical_m}"
        )
    if logical_k < 8 or logical_k % 8 != 0:
        raise ValueError(
            f"SM80 TF32 MMA requires BLK_K to be a positive multiple of 8, got {logical_k}"
        )
    if tile_n < 8 or tile_n % 8 != 0:
        raise ValueError(f"tile_n must be a positive multiple of 8, got {tile_n}")
    if max_tiles_per_execution_segment < 1:
        raise ValueError("max_tiles_per_execution_segment must be positive")

    physical_m = ((logical_m + 15) // 16) * 16
    notes = [
        "Format conversion, optional row reorder, and output inverse mapping are outside timing.",
        "All generated GPU work, including work on physically padded zeros, is inside timing.",
    ]
    if logical_m < physical_m:
        notes.append(
            f"Each logical {logical_m}-row condensation window is zero-padded to the "
            f"SM80 MMA/store tile of {physical_m} rows."
        )
    if row_reorder == "row_rcm":
        notes.append(
            "Apply deterministic Reverse Cuthill-McKee to the symmetrized structural "
            "row graph, then restore original row order after the kernel."
        )
    elif row_reorder == "first_col_sort":
        notes.append(
            "Sort rows deterministically by first structural column, then restore original "
            "output row order after the kernel."
        )
    elif row_reorder == "jaccard_greedy":
        notes.append(
            "Apply deterministic scalable minhash-bucketed greedy Jaccard row ordering, "
            "then restore original output row order after the kernel."
        )
    physical_k = logical_k
    if packing_mode == "active_fixed_blocks":
        # The BIT-BSR restore/load path is built around one SM80 TF32 K=8
        # micro-tile.  A wider logical format block is represented by adjacent
        # physical micro-blocks; this is an exact refinement of the format and
        # avoids asking the restore swizzle to span multiple MMA-K tiles.
        physical_k = 8
        notes.append(
            "Refine each logical BLK_M-by-BLK_K block into active BLK_M-by-8 "
            "SM80 TF32 micro-blocks; val_sidx contains one global physical "
            "micro-block column index per active block."
        )
    return BackendPlan(
        backend=backend,
        codegen_format_name=codegen_format_name,
        target="sm80",
        compute_capability=80,
        logical_blk_m=logical_m,
        logical_blk_k=logical_k,
        physical_blk_m=physical_m,
        physical_blk_k=physical_k,
        tile_n=tile_n,
        mma_m=16,
        mma_n=8,
        mma_k=8,
        max_tiles_per_execution_segment=max_tiles_per_execution_segment,
        packing_mode=packing_mode,
        row_reorder=row_reorder,
        notes=tuple(notes),
    )
