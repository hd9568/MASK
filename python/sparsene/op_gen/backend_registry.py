"""Semantic Format-DSL to executable-backend selection.

Backend matching is based on hierarchy/atomic semantics, never on a dataset or
an LLM-provided format name.  A backend plan also makes explicit any legal
physical padding needed by the target hardware.
"""

from __future__ import annotations

from copy import deepcopy
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
        block = _axis_map(levels[0])
        x_outer = _expect_axis(block, "X_o", "df", "ROW")
        _expect_axis(block, "Y_o_s", "sv", "COL")
        _expect_axis(block, "X_i", "df", "ROW")
        _expect_axis(block, "Y_i", "df", "COL")
        if atomic.get("x_axis") != "X_i" or atomic.get("y_axis") != "Y_i":
            raise ValueError("active fixed-block leaf must map x_axis=X_i and y_axis=Y_i")
        if atomic_type == "dense":
            # The semantic format stores every active fixed column block as a
            # dense tile.  The physical ABI expands its block-column id into
            # BLK_K consecutive scalar columns and then reuses the generated
            # SR-BCRS dense Tensor-Core data path.  This is an exact metadata
            # refinement, not a replacement of the Agent format.
            packing_mode = "active_fixed_blocks_dense"
            backend = "bit_bsr_dense_tensorcore_v1"
            codegen_format_name = "SR_BCRS"
        else:
            packing_mode = "active_fixed_blocks"
            backend = "bit_bsr_mco_tensorcore_v1"
            codegen_format_name = "BIT_BSR"
    elif len(levels) == 3:
        # Sparse-row SR-BCRS: the outer sparse row domain removes structurally
        # empty rows; the remaining two levels are ordinary column-condensed
        # dense Tensor-Core tiles.  The host bridge preserves an inverse row
        # map and leaves omitted output rows at zero.
        sparse_rows = _axis_map(levels[0])
        outer = _axis_map(levels[1])
        inner = _axis_map(levels[2])
        x_outer = _expect_axis(sparse_rows, "X_s", "sv", "ROW")
        _expect_axis(sparse_rows, "Y", "df", "COL")
        _expect_axis(outer, "X_so", "df", "ROW")
        _expect_axis(outer, "Y_s", "sv", "COL")
        _expect_axis(outer, "X_si", "df", "ROW")
        _expect_axis_type_in(inner, "Y_so", {"df", "dv"}, "COL")
        _expect_axis(inner, "Y_si", "df", "COL")
        _expect_axis(inner, "X_si", "df", "ROW")
        if atomic_type != "dense":
            raise ValueError("sparse-row column-condensed backend requires a dense atomic leaf")
        if atomic.get("x_axis") != "X_si" or atomic.get("y_axis") != "Y_si":
            raise ValueError(
                "sparse-row dense leaf must map x_axis=X_si and y_axis=Y_si"
            )
        packing_mode = "sparse_row_column_condensed_dense"
        backend = "sparse_row_sr_bcrs_dense_tensorcore_v1"
        codegen_format_name = "SR_BCRS"
    else:
        raise ValueError(
            "No executable backend for this hierarchy: expected two-level column "
            "condensation, one-level active fixed blocks, or sparse-row SR-BCRS"
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
        raise ValueError("row reorder must be attached to the outer sparse/dense row axis")
    if packing_mode == "sparse_row_column_condensed_dense" and row_reorder:
        raise ValueError(
            "sparse-row physicalization does not yet compose with an additional row reorder"
        )

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
    elif packing_mode == "active_fixed_blocks_dense":
        notes.append(
            "Expand each active fixed block-column id into BLK_K consecutive scalar "
            "columns for the generated dense SR-BCRS ABI; tile values and arithmetic "
            "remain exactly those of the Agent-designed dense fixed-block format."
        )
    elif packing_mode == "sparse_row_column_condensed_dense":
        notes.append(
            "Pack only structurally active rows into the physical row domain, retain "
            "an inverse row map, and materialize omitted output rows as exact zeros."
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


def materialize_backend_format_spec(
    spec: Mapping[str, Any], plan: BackendPlan
) -> dict[str, Any]:
    """Return the explicit Format DSL consumed by the generated-kernel ABI.

    Most backends lower the Agent proposal directly.  Two semantically richer
    profiles use a documented physical view of the same data so that the
    already validated SR-BCRS middle end can generate their dense Tensor-Core
    compute body.  The original immutable artifact remains the source format
    and is recorded separately in every generation manifest.
    """
    physical = deepcopy(dict(spec))
    if plan.packing_mode not in {
        "active_fixed_blocks_dense",
        "sparse_row_column_condensed_dense",
    }:
        return physical

    blk_m = int(plan.logical_blk_m)
    blk_k = int(plan.logical_blk_k)
    physical["format_name"] = f"{spec['format_name']}.physical_sr_bcrs"
    physical["levels"] = [
        {
            "level": 1,
            "name": "physical_column_condensation",
            "axes": [
                {
                    "name": "X_o",
                    "type": "df",
                    "direction": "ROW",
                    "length": "M",
                    "reorder": None,
                    "purpose": "physical row windows for the generated SR-BCRS ABI",
                },
                {
                    "name": "Y_s",
                    "type": "sv",
                    "direction": "COL",
                    "length": "K",
                    "reorder": None,
                    "purpose": "physical scalar-column sequence per row window",
                },
                {
                    "name": "X_i",
                    "type": "df",
                    "direction": "ROW",
                    "length": "BLK_M",
                    "reorder": None,
                    "purpose": "physical Tensor-Core tile rows",
                },
            ],
        },
        {
            "level": 2,
            "name": "physical_tensorcore_block",
            "axes": [
                {
                    "name": "Y_so",
                    "type": "df",
                    "direction": "COL",
                    "length": "K/BLK_K",
                    "reorder": None,
                    "purpose": "physical dense tile index",
                },
                {
                    "name": "Y_si",
                    "type": "df",
                    "direction": "COL",
                    "length": "BLK_K",
                    "reorder": None,
                    "purpose": "physical Tensor-Core tile columns",
                },
                {
                    "name": "X_i",
                    "type": "df",
                    "direction": "ROW",
                    "length": "BLK_M",
                    "reorder": None,
                    "purpose": "inherited physical tile rows",
                },
            ],
        },
    ]
    physical["atomic"] = {
        "type": "dense",
        "x_axis": "X_i",
        "y_axis": "Y_si",
        "layout": "row_major",
        "rationale": (
            "backend physical view for the generated SR-BCRS dense Tensor-Core ABI"
        ),
        "custom_spec": None,
    }
    physical["params"] = {"BLK_M": blk_m, "BLK_K": blk_k}
    return physical
