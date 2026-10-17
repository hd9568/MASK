from __future__ import annotations

import unittest

from run_format_design_batch import (
    _alternative_is_complete,
    _is_retryable_transport_failure,
    _quality_issues,
    _quality_warnings,
)


class FormatDesignBatchQualityTest(unittest.TestCase):
    def test_retries_transient_transport_fallbacks_only(self) -> None:
        from types import SimpleNamespace

        def response(reason: str):
            return SimpleNamespace(
                agent_metadata={"source": "fallback"},
                rationale=reason,
            )

        self.assertTrue(
            _is_retryable_transport_failure(
                response("Agent call failed: SSE ended before a terminal [DONE]")
            )
        )
        self.assertTrue(
            _is_retryable_transport_failure(
                response("Agent call failed: openai_compatible could not reach endpoint: SSL EOF")
            )
        )
        self.assertFalse(
            _is_retryable_transport_failure(
                response("Agent call failed with status=401: Invalid token")
            )
        )
        self.assertFalse(_is_retryable_transport_failure(response("JSON parse failed")))

    def test_missing_chinese_rationale_is_a_warning_not_a_quality_gate(self) -> None:
        from types import SimpleNamespace
        from unittest.mock import patch

        response = SimpleNamespace(
            agent_metadata={"source": "model", "run_dir": "/tmp/agent"},
            dsl_validation={"status": "valid", "errors": []},
            feature_evidence=[{}] * 6,
            design_summary="有效的中文设计摘要",
            format_family="column_condensed_dense_blocks",
            rationale_cn="",
            format_semantics_cn=["语义一", "语义二"],
            alternatives_considered=[
                {
                    "name": f"alternative-{index}",
                    "why_rejected": "不如所选方案",
                    "structure": "行窗口 + dense leaf",
                    "pros": ["有明确复用"],
                    "cons": ["存在块内填充"],
                }
                for index in range(3)
            ],
            transformation_plan=[
                {"op": "x", "target": "y", "result": "z", "reason": "r"}
            ] * 2,
            levels=[{"name": "L", "axes": [{"name": "X", "purpose": "axis"}]}],
            atomic={"rationale": "atomic rationale"},
            selection_reason_cn="选择理由",
            implementation_requirements=["实现要求"],
            risks_cn=["风险"],
            capability={"rts_derivable": True},
        )
        with patch("run_format_design_batch.build_format_from_response", return_value=object()), patch(
            "run_format_design_batch.derive_rts", return_value=SimpleNamespace(sequence=[1])
        ):
            issues = _quality_issues(response)
        self.assertNotIn("missing rationale_cn", issues)
        self.assertIn("missing rationale_cn (non-blocking; DSL semantics are unaffected)", _quality_warnings(response))

    def test_accepts_probe_style_detailed_alternative(self) -> None:
        alternative = {
            "name": "TCF16x8_MCO",
            "expected_rank": 2,
            "levels": [{"level": 1, "axes": [{"name": "X", "direction": "ROW"}]}],
            "atomic": {"type": "mco"},
            "semantics_cn": "行窗口凝聚列后使用 MCO 叶子。",
            "physical_work_ledger": {
                "mma_padding": "完整执行物理 tile。",
                "b_gathers_reuse": "窗口内复用 B 列。",
                "a_metadata_restore": "读取 mask 并恢复 A tile。",
                "tail_reduction_sync": "长窗口需要分段归约。",
            },
            "why_rejected": "保留为 fallback，实际排名需要实测。",
        }
        self.assertTrue(_alternative_is_complete(alternative))

    def test_rejects_detailed_alternative_missing_physical_cost(self) -> None:
        alternative = {
            "name": "TCF16x8_MCO",
            "levels": [{"level": 1}],
            "atomic": {"type": "mco"},
            "semantics_cn": "行窗口凝聚列后使用 MCO 叶子。",
            "physical_work_ledger": {"mma_padding": "完整执行物理 tile。"},
            "why_rejected": "保留为 fallback。",
        }
        self.assertFalse(_alternative_is_complete(alternative))

    def test_accepts_classic_structure_pros_cons_alternative(self) -> None:
        alternative = {
            "name": "Dense blocks",
            "structure": "行窗口 + dense atomic",
            "pros": ["省去 mask 解码"],
            "cons": ["存储并计算块内零值"],
            "why_rejected": "块密度低于 MCO。",
        }
        self.assertTrue(_alternative_is_complete(alternative))


if __name__ == "__main__":
    unittest.main()
