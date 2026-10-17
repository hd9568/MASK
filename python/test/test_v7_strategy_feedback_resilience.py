from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from run_v7_strategy_feedback import (
    _call_strategy_with_retry,
    _normalize_v7_strategy,
    _v7_strategy_search_space,
)
from sparsene.op_gen.strategy_agent import StrategyDecision


class V7StrategyFeedbackResilienceTest(unittest.TestCase):
    def test_v7_search_space_excludes_unimplemented_b_load_modes(self):
        space = _v7_strategy_search_space()
        tensor = space["tensor_placement"]
        self.assertEqual(tensor["materialize_b_array_ref_to_smem"], [True])
        self.assertEqual(tensor["enable_b_array_ref_cp_async"], [True])
        self.assertEqual(tensor["b_array_ref_mode"], ["cp_async_1x4"])
        self.assertEqual(space["pipeline"]["enable_async_pipeline"], [True])
        self.assertIn(
            "v7 semantic B-load fusion",
            space["field_catalog"]["tensor_placement"][
                "materialize_b_array_ref_to_smem"
            ]["applicability"],
        )

    def test_v7_normalizer_restores_fused_b_load_contract(self):
        proposal = StrategyDecision.from_dict(
            {
                "provider": "test",
                "rationale": "try direct B loads",
                "tensor_placement": {
                    "materialize_b_array_ref_to_smem": False,
                    "enable_b_array_ref_cp_async": False,
                    "b_array_ref_mode": "sync_scalar",
                },
                "pipeline": {"enable_async_pipeline": False},
            }
        )
        normalized = _normalize_v7_strategy(proposal, packing_mode="column_condensed")
        tensor = normalized.tensor_placement
        self.assertTrue(tensor.materialize_b_array_ref_to_smem)
        self.assertTrue(tensor.enable_b_array_ref_cp_async)
        self.assertEqual(tensor.b_array_ref_mode, "cp_async_1x4")
        self.assertTrue(normalized.pipeline.enable_async_pipeline)
        audit = normalized.agent_metadata["backend_normalization"]["v7_b_load_contract"]
        self.assertIs(
            audit["agent_values_before_normalization"]["tensor_placement"][
                "materialize_b_array_ref_to_smem"
            ],
            False,
        )

    def test_strategy_request_retries_transient_transport_error_and_audits(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            audit_path = Path(temporary_dir) / "strategy_request_retries.json"
            calls = 0

            def request():
                nonlocal calls
                calls += 1
                if calls == 1:
                    raise RuntimeError(
                        "openai_compatible could not reach endpoint: "
                        "[SSL: UNEXPECTED_EOF_WHILE_READING]"
                    )
                return "ok"

            with patch("run_v7_strategy_feedback.time.sleep") as sleep:
                result = _call_strategy_with_retry(request, audit_path=audit_path)

            self.assertEqual(result, "ok")
            self.assertEqual(calls, 2)
            sleep.assert_called_once_with(2.0)
            audit = json.loads(audit_path.read_text())
            self.assertEqual(audit["status"], "succeeded_after_retry")
            self.assertEqual(
                [item["status"] for item in audit["attempts"]], ["failed", "succeeded"]
            )

    def test_strategy_request_does_not_retry_nontransient_failure(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            audit_path = Path(temporary_dir) / "strategy_request_retries.json"
            calls = 0

            def request():
                nonlocal calls
                calls += 1
                raise ValueError("invalid strategy JSON")

            with patch("run_v7_strategy_feedback.time.sleep") as sleep:
                with self.assertRaisesRegex(ValueError, "invalid strategy JSON"):
                    _call_strategy_with_retry(request, audit_path=audit_path)

            self.assertEqual(calls, 1)
            sleep.assert_not_called()
            self.assertEqual(json.loads(audit_path.read_text())["status"], "failed")


if __name__ == "__main__":
    unittest.main()
