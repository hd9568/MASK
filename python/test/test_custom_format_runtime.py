from __future__ import annotations

import json
from pathlib import Path

import pytest

from sparsene.formats.custom.loader import load_custom_format
from sparsene.formats.dsl import build_format_from_spec, validate_format_spec
from sparsene.op_gen.backend_registry import (
    materialize_backend_format_spec,
    select_backend,
)
from sparsene.op_gen.format_portfolio import (
    executable_atomic_portfolio,
    executable_tcf_portfolio,
)
from sparsene.transform.rts import derive_rts


REPO_ROOT = Path(__file__).resolve().parents[2]
CUSTOM_DIR = REPO_ROOT / "python/sparsene/formats/custom"


def test_all_exported_v6_formats_are_importable_and_rts_derivable() -> None:
    index = json.loads((CUSTOM_DIR / "index.json").read_text(encoding="utf-8"))
    assert index["run_id"] == "20260718_165146_deepseek_v4_pro_format26_v6"
    assert len(index["datasets"]) == 26
    for dataset in index["datasets"]:
        spec, format_obj, _module = load_custom_format(dataset)
        assert validate_format_spec(spec)["status"] == "valid"
        assert derive_rts(format_obj).sequence


@pytest.mark.parametrize(
    (
        "dataset",
        "codegen_format",
        "logical_m",
        "logical_k",
        "physical_m",
        "physical_k",
        "reorder",
        "packing_mode",
    ),
    [
        ("ASIC_680k", "BIT_TCF", 8, 16, 16, 16, None, "column_condensed"),
        ("DD", "BIT_TCF", 16, 16, 16, 16, "row_rcm", "column_condensed"),
        ("Ga41As41H72", "BIT_TCF", 8, 32, 16, 32, None, "column_condensed"),
        ("OVCAR-8H", "BIT_TCF", 16, 8, 16, 8, None, "column_condensed"),
        ("Si41Ge41H72", "BIT_TCF", 8, 32, 16, 32, None, "column_condensed"),
        ("Yeast", "BIT_TCF", 16, 8, 16, 8, "first_col_sort", "column_condensed"),
        ("YeastH", "BIT_TCF", 32, 8, 32, 8, None, "column_condensed"),
        ("cant", "BIT_TCF", 32, 32, 32, 32, None, "column_condensed"),
        ("conf5_4-8x8-10", "BIT_TCF", 16, 16, 16, 16, None, "column_condensed"),
        ("consph", "BIT_TCF", 16, 8, 16, 8, None, "column_condensed"),
        ("cop20k_A", "BIT_TCF", 16, 8, 16, 8, None, "column_condensed"),
        ("dc2", "BIT_BSR", 16, 16, 16, 8, None, "active_fixed_blocks"),
        ("ddi", "BIT_TCF", 16, 16, 16, 16, "first_col_sort", "column_condensed"),
        ("eu-2005", "BIT_TCF", 16, 16, 16, 16, None, "column_condensed"),
        ("mac_econ_fwd500", "BIT_TCF", 16, 16, 16, 16, None, "column_condensed"),
        ("mc2depi", "BIT_TCF", 16, 32, 16, 32, None, "column_condensed"),
        ("mip1", "BIT_TCF", 8, 16, 16, 16, None, "column_condensed"),
        ("pdb1HYS", "BIT_TCF", 32, 16, 32, 16, None, "column_condensed"),
        ("protein", "BIT_TCF", 16, 16, 16, 16, None, "column_condensed"),
        ("pwtk", "BIT_TCF", 16, 16, 16, 16, None, "column_condensed"),
        ("reddit", "BIT_TCF", 16, 32, 16, 32, None, "column_condensed"),
        ("rma10", "BIT_TCF", 16, 16, 16, 16, None, "column_condensed"),
        ("scircuit", "BIT_TCF", 16, 16, 16, 16, None, "column_condensed"),
        ("shipsec1", "BIT_TCF", 16, 8, 16, 8, None, "column_condensed"),
        ("web-BerkStan", "BIT_TCF", 16, 8, 16, 8, "jaccard_greedy", "column_condensed"),
        ("webbase-1M", "BIT_TCF", 16, 16, 16, 16, None, "column_condensed"),
    ],
)
def test_all_v6_formats_select_expected_semantic_backend(
    dataset: str,
    codegen_format: str,
    logical_m: int,
    logical_k: int,
    physical_m: int,
    physical_k: int,
    reorder: str | None,
    packing_mode: str,
) -> None:
    spec, _format_obj, _module = load_custom_format(dataset)
    plan = select_backend(spec)
    assert plan.codegen_format_name == codegen_format
    assert plan.logical_blk_m == logical_m
    assert plan.logical_blk_k == logical_k
    assert plan.physical_blk_m == physical_m
    assert plan.physical_blk_k == physical_k
    assert plan.row_reorder == reorder
    assert plan.packing_mode == packing_mode


def test_backend_selection_does_not_depend_on_format_name() -> None:
    spec, _format_obj, _module = load_custom_format("ASIC_680k")
    renamed = dict(spec)
    renamed["format_name"] = "A_completely_new_research_name"
    assert select_backend(renamed) == select_backend(spec)


def test_frontend_csr_is_explicitly_physicalized_to_canonical_tcf() -> None:
    spec, _format_obj, _module = load_custom_format("ASIC_680k")
    spec = json.loads(json.dumps(spec))
    spec["atomic"]["type"] = "csr"
    plan = select_backend(spec)
    assert plan.backend == "frontend_canonical_tcf_mco_tensorcore_v1"
    assert plan.packing_mode == "column_condensed"
    physical = materialize_backend_format_spec(spec, plan)
    assert physical != spec
    assert physical["atomic"]["type"] == "mco"
    assert validate_format_spec(physical)["status"] == "valid"
    assert derive_rts(build_format_from_spec(physical)).sequence
    assert select_backend(physical).backend == "tcf_mco_tensorcore_v1"


def test_atomic_portfolio_preserves_agent_dsl_and_adds_only_semantic_sibling() -> None:
    spec, _format_obj, _module = load_custom_format("ASIC_680k")
    original_atomic = dict(spec["atomic"])
    candidates = executable_atomic_portfolio(spec)
    assert [candidate.candidate_id for candidate in candidates] == [
        "agent",
        "atomic_dense_sibling",
    ]
    assert candidates[0].source == "agent_proposal"
    assert candidates[0].backend_plan.codegen_format_name == "BIT_TCF"
    assert candidates[1].source == "deterministic_atomic_sibling"
    assert candidates[1].atomic_type == "dense"
    assert candidates[1].backend_plan.codegen_format_name == "SR_BCRS"
    assert spec["atomic"] == original_atomic
    assert candidates[0].spec_sha256 != candidates[1].spec_sha256


def test_atomic_portfolio_does_not_invent_an_unimplemented_bsr_dense_sibling() -> None:
    spec, _format_obj, _module = load_custom_format("dc2")
    candidates = executable_atomic_portfolio(spec)
    assert len(candidates) == 1
    assert candidates[0].candidate_id == "agent"
    assert candidates[0].backend_plan.codegen_format_name == "BIT_BSR"


def test_tcf_portfolio_adds_hardware_aligned_dense_tiles_without_mutating_agent_spec() -> None:
    spec, _format_obj, _module = load_custom_format("ASIC_680k")
    candidates = executable_tcf_portfolio(spec)
    dense_shapes = {
        (candidate.spec["params"]["BLK_M"], candidate.spec["params"]["BLK_K"])
        for candidate in candidates
        if candidate.atomic_type == "dense"
    }
    assert {(8, 16), (16, 8), (16, 16), (32, 8)}.issubset(dense_shapes)
    assert spec["params"] == {"BLK_M": 8, "BLK_K": 16}


def _dense_profile_spec(levels: list[dict], atomic_axes: tuple[str, str]) -> dict:
    return {
        "schema_version": "sparsene.format_agent.v7",
        "dataset": "semantic-profile-test",
        "format_name": "semantic_profile_test",
        "levels": levels,
        "atomic": {
            "type": "dense",
            "x_axis": atomic_axes[0],
            "y_axis": atomic_axes[1],
            "layout": "row_major",
        },
        "params": {"BLK_M": 16, "BLK_K": 8},
    }


def test_dense_active_fixed_blocks_selects_explicit_physical_backend() -> None:
    spec = _dense_profile_spec(
        [
            {
                "level": 1,
                "name": "active_blocks",
                "axes": [
                    {"name": "X_o", "type": "df", "direction": "ROW", "length": "M"},
                    {"name": "Y_o_s", "type": "sv", "direction": "COL", "length": "K"},
                    {"name": "X_i", "type": "df", "direction": "ROW", "length": "BLK_M"},
                    {"name": "Y_i", "type": "df", "direction": "COL", "length": "BLK_K"},
                ],
            }
        ],
        ("X_i", "Y_i"),
    )
    plan = select_backend(spec)
    assert plan.packing_mode == "active_fixed_blocks_dense"
    assert plan.codegen_format_name == "SR_BCRS"
    physical = materialize_backend_format_spec(spec, plan)
    assert physical != spec
    assert validate_format_spec(physical)["status"] == "valid"
    assert derive_rts(build_format_from_spec(physical)).sequence


def test_sparse_row_dense_sr_bcrs_selects_explicit_physical_backend() -> None:
    spec = _dense_profile_spec(
        [
            {
                "level": 1,
                "name": "sparse_row_domain",
                "axes": [
                    {"name": "X_s", "type": "sv", "direction": "ROW", "length": "M"},
                    {"name": "Y", "type": "df", "direction": "COL", "length": "K"},
                ],
            },
            {
                "level": 2,
                "name": "sparse_row_blocks",
                "axes": [
                    {"name": "X_so", "type": "df", "direction": "ROW", "length": "M/BLK_M"},
                    {"name": "Y_s", "type": "sv", "direction": "COL", "length": "K"},
                    {"name": "X_si", "type": "df", "direction": "ROW", "length": "BLK_M"},
                ],
            },
            {
                "level": 3,
                "name": "tensorcore_block",
                "axes": [
                    {"name": "Y_so", "type": "df", "direction": "COL", "length": "K/BLK_K"},
                    {"name": "Y_si", "type": "df", "direction": "COL", "length": "BLK_K"},
                    {"name": "X_si", "type": "df", "direction": "ROW", "length": "BLK_M"},
                ],
            },
        ],
        ("X_si", "Y_si"),
    )
    plan = select_backend(spec)
    assert plan.packing_mode == "sparse_row_column_condensed_dense"
    assert plan.codegen_format_name == "SR_BCRS"
    physical = materialize_backend_format_spec(spec, plan)
    assert physical != spec
    assert validate_format_spec(physical)["status"] == "valid"
    assert derive_rts(build_format_from_spec(physical)).sequence


def test_heavy_row_partition_frontend_falls_back_transparently_to_canonical_tcf() -> None:
    spec = {
        "schema_version": "sparsene.format_design.v6",
        "dataset": "heavy-row-partition-test",
        "format_name": "HeavyRowPartitioned_SegmentedTCF_MCO_16x8",
        "levels": [
            {
                "level": 1,
                "name": "heavy_rows",
                "axes": [
                    {"name": "X_o", "type": "df", "direction": "ROW", "length": "M",
                     "reorder": "stable_heavy_row_partition"},
                    {"name": "Y_s", "type": "sv", "direction": "COL", "length": "K"},
                    {"name": "X_i", "type": "df", "direction": "ROW", "length": "BLK_M"},
                ],
            },
            {
                "level": 2,
                "name": "segmented_columns",
                "axes": [
                    {"name": "Y_go", "type": "dv", "direction": "COL", "length": "K/SEG_K"},
                    {"name": "Y_gi", "type": "df", "direction": "COL", "length": "SEG_K"},
                    {"name": "X_i", "type": "df", "direction": "ROW", "length": "BLK_M"},
                ],
            },
            {
                "level": 3,
                "name": "tensorcore_tile",
                "axes": [
                    {"name": "Y_so", "type": "dv", "direction": "COL", "length": "SEG_K/BLK_K"},
                    {"name": "Y_si", "type": "df", "direction": "COL", "length": "BLK_K"},
                    {"name": "X_i", "type": "df", "direction": "ROW", "length": "BLK_M"},
                ],
            },
        ],
        "atomic": {"type": "mco", "x_axis": "X_i", "y_axis": "Y_si", "layout": "row_major"},
        "params": {"BLK_M": 16, "BLK_K": 8, "SEG_K": 256},
    }
    plan = select_backend(spec)
    assert plan.backend == "frontend_canonical_tcf_mco_tensorcore_v1"
    assert plan.row_reorder is None
    assert any("does not realize the proposed heavy-row partition or segmentation" in note
               for note in plan.notes)
    physical = materialize_backend_format_spec(spec, plan)
    assert physical != spec
    assert validate_format_spec(physical)["status"] == "valid"
    assert derive_rts(build_format_from_spec(physical)).sequence
    assert select_backend(physical).backend == "tcf_mco_tensorcore_v1"
