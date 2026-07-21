from __future__ import annotations

import json
from pathlib import Path

import pytest

from sparsene.formats.custom.loader import load_custom_format
from sparsene.formats.dsl import validate_format_spec
from sparsene.op_gen.backend_registry import select_backend
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
