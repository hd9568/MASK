from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from sparsene.agent.client import TokenUsage
from sparsene.op_gen.format_design_agent import (
    DESIGN_PROFILE,
    DESIGN_SCHEMA_VERSION,
    FormatDesignRequest,
    FormatDesignResponse,
    FormatDesignConfig,
    _canonical_atomic,
    _canonical_levels,
    _default_output_schema,
    assess_backend_capability,
    build_format_from_response,
    normalize_format_payload,
    render_format_design_prompt,
    run_format_design_agent,
    validate_format_dsl,
)
from sparsene.performance_contract import (
    EVALUATION_CONTRACT_VERSION,
    KERNEL_ONLY_EVALUATION_CONTRACT,
)
from sparsene.formats.design_catalog import (
    SOURCE_FORMAT_CATALOG,
    complete_source_designs,
)
from sparsene.transform.rts import derive_rts


def _features() -> dict:
    return {
        "schema_version": "sparsene.matrix_features.v3",
        "shape": {"M": 64, "K": 128, "nnz": 100},
        "density": 0.0122,
        "sparsity": 0.9878,
        "data_quality": {"invalid_entries": 0},
        "row_nnz_distribution": {
            "mean": 1.56,
            "p90": 4,
            "cv": 1.2,
            "gini": 0.62,
            "zero_fraction": 0.25,
        },
        "col_nnz_distribution": {
            "mean": 0.78,
            "p90": 2,
            "cv": 1.5,
            "gini": 0.7,
            "zero_fraction": 0.4,
        },
        "locality": {
            "normalized_bandwidth": 0.18,
            "diagonal_band_fraction_1pct": 0.31,
            "row_span_distribution": {"mean": 23.0, "p90": 50.0, "cv": 0.7},
            "column_gap_distribution": {"mean": 4.0, "p90": 9.0, "cv": 1.1},
        },
        "reorder_proxy": {
            "adjacent_row_jaccard": 0.22,
            "first_col_sorted_jaccard": 0.39,
            "jaccard_gain": 0.17,
            "bm16_unique_col_reduction": 0.28,
        },
        "hierarchical_candidates": {
            "16x8": {
                "blk_m": 16,
                "blk_k": 8,
                "total_tc_blocks": 20,
                "coo_padded_nnz": 112,
                "unique_columns_total": 80,
                "b_index_padding_ratio": 0.0,
                "tensorcore_zero_fraction": 0.72,
                "structural_density": 0.28,
                "tc_blocks_per_window": {"mean": 5, "cv": 1.0},
                "structural_nnz_per_tc_block": {"mean": 6, "p90": 13, "cv": 1.3},
                "storage_estimate_bytes": {
                    "coo": 1800,
                    "mask": 1200,
                    "dense": 3000,
                    "smallest": "mask",
                },
            },
            "16x16": {
                "blk_m": 16,
                "blk_k": 16,
                "total_tc_blocks": 10,
                "coo_padded_nnz": 120,
                "unique_columns_total": 80,
                "b_index_padding_ratio": 0.1,
                "tensorcore_zero_fraction": 0.76,
                "structural_density": 0.24,
                "tc_blocks_per_window": {"mean": 2.5, "cv": 0.8},
                "structural_nnz_per_tc_block": {"mean": 10, "p90": 20, "cv": 1.1},
                "storage_estimate_bytes": {
                    "coo": 1700,
                    "mask": 1000,
                    "dense": 3500,
                    "smallest": "mask",
                },
            },
        },
    }


def _novel_payload() -> dict:
    return {
        "schema_version": DESIGN_SCHEMA_VERSION,
        "format_name": "ReorderedHierarchicalMask",
        "format_family": "hybrid",
        "design_summary": "Row-window condensation plus an inner sparse sub-tile and MCO leaf.",
        "levels": [
            {
                "level": 1,
                "name": "reordered_row_window",
                "axes": [
                    {
                        "name": "X_o",
                        "type": "df",
                        "direction": "ROW",
                        "length": "M",
                        "reorder": "row_similarity",
                        "purpose": "group similar rows",
                    },
                    {
                        "name": "Y_s",
                        "type": "sv",
                        "direction": "COL",
                        "length": "K",
                        "reorder": None,
                        "purpose": "condense touched columns",
                    },
                    {
                        "name": "X_i",
                        "type": "df",
                        "direction": "ROW",
                        "length": "BLK_M",
                        "reorder": None,
                        "purpose": "row window extent",
                    },
                ],
            },
            {
                "level": 2,
                "name": "tc_block",
                "axes": [
                    {
                        "name": "Y_so",
                        "type": "df",
                        "direction": "COL",
                        "length": "K/BLK_K",
                        "reorder": None,
                        "purpose": "outer condensed block",
                    },
                    {
                        "name": "Y_si",
                        "type": "sf",
                        "direction": "COL",
                        "length": "BLK_K",
                        "reorder": None,
                        "purpose": "sparse active columns",
                    },
                    {
                        "name": "X_i",
                        "type": "df",
                        "direction": "ROW",
                        "length": "BLK_M",
                        "reorder": None,
                        "purpose": "tile rows",
                    },
                ],
            },
        ],
        "atomic": {
            "type": "mco",
            "x_axis": "X_i",
            "y_axis": "Y_si",
            "layout": "row_major",
            "rationale": "mask storage is estimated smaller than COO and dense",
        },
        "params": {"BLK_M": 16, "BLK_K": 8},
        "selected_candidate": "16x8",
        "transformation_plan": [
            {
                "op": "reorder",
                "target": "X_o",
                "result": "similar rows become adjacent",
                "reason": "jaccard_gain is positive",
            },
            {
                "op": "atomicize",
                "target": "X_i,Y_si",
                "result": "mask plus compact values",
                "reason": "mask storage estimate is lowest",
            },
        ],
        "feature_evidence": [
            {"feature": "row_nnz.cv", "value": 1.2, "implication": "rows are skewed"},
            {"feature": "col_nnz.gini", "value": 0.7, "implication": "columns are skewed"},
            {
                "feature": "reorder_proxy.jaccard_gain",
                "value": 0.17,
                "implication": "row grouping may improve overlap",
            },
            {
                "feature": "hierarchical_candidates.16x8.structural_density",
                "value": 0.28,
                "implication": "dense leaves waste too much work",
            },
            {
                "feature": "hierarchical_candidates.16x8.storage_estimate_bytes.mask",
                "value": 1200,
                "implication": "mask is compact",
            },
            {
                "feature": "hierarchical_candidates.16x8.storage_estimate_bytes.coo",
                "value": 1800,
                "implication": "COO metadata is larger",
            },
        ],
        "alternatives_considered": [
            {
                "name": "COO-TCF",
                "structure": "two-level TCF + COO",
                "pros": ["handles low density"],
                "cons": ["more coordinate bytes"],
                "why_rejected": "mask estimate is lower",
            },
            {
                "name": "Dense-block",
                "structure": "row windows + dense leaf",
                "pros": ["simple restore"],
                "cons": ["72% zero work"],
                "why_rejected": "block density is too low",
            },
            {
                "name": "Mask-no-reorder",
                "structure": "TCF + MCO without reorder",
                "pros": ["no preprocessing permutation"],
                "cons": ["misses overlap gain"],
                "why_rejected": "post-reorder kernel proxy predicts better B reuse",
            },
        ],
        "tradeoffs": ["Reordering may improve timed-kernel reuse; its preprocessing and inverse mapping are untimed engineering work."],
        "rationale": "Skew, positive reorder gain, and mask storage favor a reordered MCO design.",
        "rationale_cn": "行列分布偏斜且重排代理有收益，块内密度不足以采用 dense，mask 估算小于 COO。",
        "decision_basis_cn": ["使用行窗口凝聚", "采用 MCO leaf", "重排仅按 kernel 收益排序"],
        "format_semantics_cn": [
            "外层先按相似性组织行并在窗口内凝聚列。",
            "内层对凝聚列分块并用 MCO 恢复 Tensor Core 逻辑 tile。",
        ],
        "selection_reason_cn": "相比 COO、dense 和无重排 mask，当前组合更贴合存储与复用特征。",
        "implementation_requirements": ["实现 row_similarity 预处理和逆映射"],
        "risks_cn": ["重排代理不能准确预测重排后的 kernel 性能。"],
        "confidence": "medium",
    }


class FormatDesignAgentTest(unittest.TestCase):
    def test_prompt_teaches_open_research_search_space(self) -> None:
        request = FormatDesignRequest(
            request_id="test",
            matrix_name="synthetic",
            matrix_features=_features(),
            hardware={"cuda_arch": "sm_80"},
            output_schema=_default_output_schema(),
            constraints=[],
            rules=[],
            empirical_feedback={
                "contract_version": "sparsene.format_design.empirical_feedback.v1",
                "prior_agent_to_dense_generated_ms_ratio": 2.087,
                "calibration_note": "directional calibration only",
            },
        )
        prompt = render_format_design_prompt(request)
        self.assertEqual(_default_output_schema()["schema_version"], DESIGN_SCHEMA_VERSION)
        self.assertIn("physical hierarchical sparse format", prompt)
        for atomic in ("coo", "mco", "csr", "ell", "dia", "dense", "custom"):
            self.assertIn(f"- {atomic}:", prompt)
        self.assertIn("EXISTING FORMATS ARE REFERENCES, NOT A CLOSED MENU", prompt)
        self.assertIn("Do not reduce the task to choosing a tile", prompt)
        self.assertIn('"level": 1', prompt)
        self.assertIn("never `\"1\"`", prompt)
        self.assertIn("FEATURE PATH-VALUE MAP", prompt)
        self.assertIn("ROW_REORDER_SR_BCRS", prompt)
        self.assertIn("BENCHMARK OBJECTIVE AND TIMING BOUNDARY", prompt)
        self.assertIn("EXECUTABLE SM80 COST MODEL FOR TCF ATOMIC COMPARISONS", prompt)
        self.assertIn("does not skip zero-position MMA work", prompt)
        self.assertIn("PRIOR EMPIRICAL FEEDBACK", prompt)
        self.assertIn("prior_agent_to_dense_generated_ms_ratio", prompt)
        self.assertIn("2.087", prompt)
        self.assertIn("outside the measured time", prompt)
        self.assertIn("do not penalize or reject a format", prompt)
        self.assertIn(EVALUATION_CONTRACT_VERSION, prompt)
        self.assertNotIn("its two-level COO DSL is immutable", prompt)

    def test_kernel_only_contract_is_deterministically_attached(self) -> None:
        normalized = normalize_format_payload(_novel_payload(), _features())
        self.assertEqual(
            normalized["evaluation_contract"], KERNEL_ONLY_EVALUATION_CONTRACT
        )
        self.assertEqual(
            normalized["evaluation_contract"]["version"],
            EVALUATION_CONTRACT_VERSION,
        )
        self.assertIn(
            "行列置换的构造与应用，以及为正确性执行的逆置换/恢复",
            normalized["evaluation_contract"]["excluded_from_primary_metric"],
        )

    def test_output_example_has_valid_json_scalar_types_and_dsl(self) -> None:
        example = _default_output_schema()
        self.assertIsInstance(example["levels"][0]["level"], int)
        self.assertTrue(all(isinstance(value, int) for value in example["params"].values()))
        validation = validate_format_dsl(
            format_name=example["format_name"],
            levels=example["levels"],
            atomic=example["atomic"],
            params=example["params"],
        )
        self.assertEqual(validation["status"], "valid", validation["errors"])

    def test_source_catalog_covers_and_validates_all_complete_declarations(self) -> None:
        expected = {
            "ME_TCF",
            "BIT_TCF_ACC",
            "CSR_TCF",
            "ELL_TCF",
            "DIA_TCF",
            "SR_BCRS",
            "BIT_BSR_SPADEN",
            "HME_TCF",
            "ROW_REORDER_SR_BCRS",
            "DENSE_AND_DENSE_T",
        }
        self.assertEqual({item["id"] for item in SOURCE_FORMAT_CATALOG}, expected)
        complete = complete_source_designs()
        self.assertEqual(len(complete), len(expected) - 1)
        for design in complete:
            with self.subTest(format_name=design["format_name"]):
                validation = validate_format_dsl(**design)
                self.assertEqual(validation["status"], "valid", validation["errors"])

    def test_numeric_json_scalars_are_canonicalized_without_changing_structure(self) -> None:
        payload = _novel_payload()
        payload["levels"][0]["level"] = "1"
        payload["levels"][1]["level"] = "2"
        payload["params"] = {"BLK_M": "16", "BLK_K": "8"}
        normalized = normalize_format_payload(payload, _features())
        self.assertEqual([level["level"] for level in normalized["levels"]], [1, 2])
        self.assertEqual(normalized["params"], {"BLK_M": 16, "BLK_K": 8})
        self.assertEqual(normalized["dsl_validation"]["status"], "valid")
        self.assertTrue(any("numeric string level" in action for action in normalized["normalization_actions"]))

    def test_truncated_response_is_explicit_failure_even_if_json_is_parseable(self) -> None:
        invocation = SimpleNamespace(
            parsed_payload=_novel_payload(),
            finish_reason="length",
            response_text='{"partial": true}',
            usage=TokenUsage(prompt_tokens=100, completion_tokens=32768, total_tokens=32868),
            run_dir=Path("/tmp/mock-format-agent"),
            resolved_model=SimpleNamespace(to_dict=lambda: {"model": "mock"}),
        )
        hardware = SimpleNamespace(
            gpu_name="mock",
            cuda_arch="sm_80",
            driver_version="mock",
            total_memory_mb=0,
            warp_size=32,
        )
        with patch(
            "sparsene.op_gen.format_design_agent.discover_hardware_profile",
            return_value=hardware,
        ), patch(
            "sparsene.op_gen.format_design_agent.invoke_json_agent",
            return_value=invocation,
        ):
            response = run_format_design_agent(
                "synthetic",
                _features(),
                FormatDesignConfig(api_key="mock", log_dir="/tmp"),
            )
        self.assertEqual(response.format_name, "FORMAT_DESIGN_FAILED")
        self.assertEqual(response.agent_metadata["source"], "fallback")
        self.assertEqual(response.agent_metadata["finish_reason"], "length")
        self.assertIn("truncated", response.rationale)

    def test_valid_novel_payload_is_preserved_and_buildable(self) -> None:
        payload = _novel_payload()
        normalized = normalize_format_payload(payload, _features())
        self.assertEqual(normalized["levels"], payload["levels"])
        self.assertEqual(normalized["atomic"], payload["atomic"])
        self.assertEqual(normalized["dsl_validation"]["status"], "valid")
        self.assertEqual(normalized["capability"]["support_level"], "frontend_research")
        self.assertEqual(normalized["normalization_actions"], [])
        response = FormatDesignResponse(request_id="test", raw_payload={}, **normalized)
        fmt = build_format_from_response(response)
        self.assertEqual(fmt.child.child.atomic_type, "mco")
        self.assertGreater(len(derive_rts(fmt).sequence), 0)

    def test_invalid_structure_is_reported_not_rewritten_to_me_tcf(self) -> None:
        payload = _novel_payload()
        payload["format_name"] = "invalid name with spaces"
        payload["levels"] = [{"level": 1, "name": "broken", "axes": []}]
        payload["atomic"] = {"type": "unknown", "x_axis": "bad", "y_axis": "bad"}
        normalized = normalize_format_payload(payload, _features())
        self.assertEqual(normalized["levels"], payload["levels"])
        self.assertEqual(normalized["atomic"], payload["atomic"])
        self.assertEqual(normalized["dsl_validation"]["status"], "invalid")
        self.assertEqual(normalized["capability"]["support_level"], "invalid")
        self.assertNotEqual(normalized["format_name"], "ME_TCF")

    def test_builder_rejects_invalid_external_response(self) -> None:
        response = FormatDesignResponse(
            request_id="test",
            format_name="bad name",
            levels=[],
            atomic={},
            params={},
            rationale="",
            confidence="low",
        )
        with self.assertRaises(ValueError):
            build_format_from_response(response)

    def test_known_me_structure_is_classified_for_integrated_e2e(self) -> None:
        validation = validate_format_dsl(
            format_name="FeatureChosenTCF",
            levels=_canonical_levels(),
            atomic=_canonical_atomic("coo"),
            params={"BLK_M": 16, "BLK_K": 8},
        )
        capability = assess_backend_capability(
            format_name="FeatureChosenTCF",
            levels=_canonical_levels(),
            atomic=_canonical_atomic("coo"),
            params={"BLK_M": 16, "BLK_K": 8},
            selected_candidate="16x8",
            validation=validation,
        )
        self.assertEqual(validation["status"], "valid")
        self.assertEqual(capability["matched_profile"], "ME_TCF")
        self.assertEqual(capability["support_level"], "integrated_e2e")
        self.assertTrue(capability["execution_ready"])

    def test_known_mask_tcf_uses_generalized_integrated_backend(self) -> None:
        capability = assess_backend_capability(
            format_name="MaskTCF",
            levels=_canonical_levels(),
            atomic=_canonical_atomic("mco"),
            params={"BLK_M": 16, "BLK_K": 8},
            selected_candidate="16x8",
        )
        self.assertEqual(capability["matched_profile"], "TCF_MCO_TENSORCORE_V1")
        self.assertEqual(capability["backend_format_name"], "BIT_TCF")
        self.assertEqual(capability["support_level"], "integrated_e2e")
        self.assertTrue(capability["execution_ready"])

    def test_fully_specified_custom_atomic_is_valid_research_design(self) -> None:
        atomic = {
            "type": "custom",
            "name": "run_length_mask",
            "x_axis": "X_i",
            "y_axis": "Y_si",
            "layout": "row_major",
            "rationale": "compress contiguous runs inside the logical tile",
            "custom_spec": {
                "value_arrays": ["values: fp32[nnz]"],
                "metadata_arrays": ["run_start: int32[nruns]", "run_len: int16[nruns]"],
                "logical_mapping": "run_start + local offset maps to (X_i,Y_si)",
                "restore_algorithm": "expand each run into the Tensor Core tile",
            },
        }
        validation = validate_format_dsl(
            format_name="RunLengthTCF",
            levels=_canonical_levels(),
            atomic=atomic,
            params={"BLK_M": 16, "BLK_K": 8},
        )
        capability = assess_backend_capability(
            format_name="RunLengthTCF",
            levels=_canonical_levels(),
            atomic=atomic,
            params={"BLK_M": 16, "BLK_K": 8},
            selected_candidate="16x8",
            validation=validation,
        )
        self.assertEqual(validation["status"], "valid")
        self.assertEqual(capability["support_level"], "frontend_research")
        self.assertTrue(capability["format_object_buildable"])
        self.assertFalse(capability["rts_derivable"])
        self.assertFalse(capability["execution_ready"])

    def test_ungrounded_evidence_is_removed_without_changing_dsl(self) -> None:
        payload = _novel_payload()
        original_levels = deepcopy(payload["levels"])
        payload["feature_evidence"][0]["value"] = 999
        payload["confidence"] = "high"
        normalized = normalize_format_payload(payload, _features())
        self.assertEqual(normalized["levels"], original_levels)
        self.assertEqual(len(normalized["feature_evidence"]), 5)
        self.assertEqual(normalized["confidence"], "medium")
        self.assertTrue(
            any("ungrounded feature evidence" in action for action in normalized["normalization_actions"])
        )


if __name__ == "__main__":
    unittest.main()
