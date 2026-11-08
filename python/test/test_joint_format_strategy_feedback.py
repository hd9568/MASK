from __future__ import annotations

from pathlib import Path
from copy import deepcopy
import csv
import json
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_joint_format_strategy_feedback import (  # noqa: E402
    _best_record,
    _compact_best_result,
    _format_config,
    _format_retry_guidance,
    _format_signature,
    _fixed_sr_bcrs_response,
    _joint_format_issues,
    _publish_dataset_summary,
    _run_fixed_initial_format,
    _run_format_agent,
    build_format_feedback,
)
from sparsene.utils.sparsene_reference import load_sparsene_reference  # noqa: E402
from run_ds_v4pro_joint_all26_campaign import DATASETS  # noqa: E402
from run_v7_strategy_feedback import _baseline_decision  # noqa: E402


class JointFormatStrategyFeedbackTest(unittest.TestCase):
    def test_all_26_historical_best_formats_have_executable_format0(self) -> None:
        variants = {}
        for dataset in DATASETS:
            reference = load_sparsene_reference(dataset)
            variants[dataset] = reference.variant
            response = _fixed_sr_bcrs_response(reference)
            self.assertEqual(response.atomic["type"], reference.atomic_type)
            self.assertTrue(response.capability["execution_ready"], dataset)
        self.assertEqual(variants["webbase-1M"], "ACC-BIT-TCF")
        self.assertEqual(variants["mip1"], "SR-BCRS-16x8-multi-bind")
        self.assertEqual(variants["ASIC_680k"], "SR-BCRS-16x8-strict-lb")

    def test_historical_reference_is_separate_from_generated_candidate(self) -> None:
        reference = load_sparsene_reference("webbase-1M")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            feature = root / "webbase-1M.features.v3.json"
            feature.write_text("{}\n", encoding="utf-8")
            record = _run_fixed_initial_format(
                dataset="webbase-1M", feature_path=feature,
                candidate_dir=root / "format_00", feedback={}, reference=reference,
            )
            self.assertEqual(record["historical_reference"]["provenance"],
                             "historical_hand_tuned_reference_not_generated")
            self.assertEqual(record["format"]["atomic"]["type"], "mco")
            self.assertEqual(record["historical_reference"]["initial_execution_policy"], "row")

    def test_historical_launch_variant_seeds_generated_heuristic(self) -> None:
        for dataset, expected in (
            ("ASIC_680k", "strict_lb"),
            ("mip1", "multi_bind"),
            ("webbase-1M", "row"),
            ("DD", "row"),
        ):
            policy = load_sparsene_reference(dataset).execution_policy
            self.assertEqual(policy, expected)
            self.assertEqual(_baseline_decision(policy).runtime_schedule.execution_policy, expected)

    def test_trajectory_keeps_historical_one_x_distinct_from_generated_iter0(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate = root / "candidate"
            iteration = candidate / "datasets" / "DD" / "strategy_tuning" / "iteration_00"
            iteration.mkdir(parents=True)
            iteration.joinpath("iteration_summary.json").write_text(json.dumps({
                "iteration": 0, "status": "validated", "operator_path": "/generated.inc",
                "e2e_result": {"timing": {"generated_ms": 2.0},
                               "correctness": {"passed": True}},
            }), encoding="utf-8")
            reference = load_sparsene_reference("DD")
            record = {
                "dataset": "DD", "format_iteration": 0, "candidate_dir": str(candidate),
                "format": {"format_name": "format0"},
                "best_generated_ms": 2.0, "best_correctness_passed": True,
                "best_strategy_iteration": 0, "best_operator_path": "/generated.inc",
                "format_artifact": "/format.json", "strategy_candidates_evaluated": 1,
            }
            args = SimpleNamespace(format_iterations=1, strategy_agent_iterations=0,
                                   initial_format="sparsene_historical_best",
                                   sparsene_reference_csv=Path(reference.source_csv))
            summary = _publish_dataset_summary(
                run_dir=root / "run", dataset="DD", records=[record], args=args,
            )
            self.assertEqual(summary["historical_sparsene_reference_speedup"], 1.0)
            self.assertAlmostEqual(summary["best_current_run_vs_historical_sparsene"],
                                   reference.milliseconds / 2.0)
            trajectory = root / "run" / "datasets" / "DD" / "performance_trajectory.csv"
            with trajectory.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["point_type"], "historical_sparsene_reference")
            self.assertEqual(rows[0]["display_iteration"], "0")
            self.assertEqual(rows[0]["speedup_vs_historical_sparsene"], "1.0")
            self.assertEqual(rows[1]["point_type"], "generated_kernel")
            self.assertEqual(rows[1]["display_iteration"], "1")
            self.assertNotEqual(rows[1]["speedup_vs_historical_sparsene"], "1.0")

    def test_fixed_sr_bcrs_format0_is_executable_and_audited_without_agent(self) -> None:
        response = _fixed_sr_bcrs_response()
        self.assertEqual(response.dsl_validation["status"], "valid")
        self.assertTrue(response.capability["execution_ready"])
        self.assertEqual(response.capability["backend_plan"]["backend"], "sr_bcrs_dense_tensorcore_v1")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            feature = root / "matrix.features.v3.json"
            feature.write_text("{}\n", encoding="utf-8")
            record = _run_fixed_initial_format(
                dataset="matrix", feature_path=feature,
                candidate_dir=root / "format_00", feedback={"outer_iteration": 0},
            )
            self.assertEqual(record["format_origin"], "fixed_repository_sr_bcrs_16x8")
            decision = (root / "format_00" / "datasets" / "matrix" / "format_decision.json").read_text()
            self.assertIn('"model_called": false', decision)
            self.assertTrue(Path(record["format_artifact"]).is_file())
            self.assertEqual(
                (root / "format_00" / "datasets" / "matrix" / "matrix_features.json").read_text(),
                "{}\n",
            )

    def test_joint_gate_rejects_backend_unsupported_reorder(self) -> None:
        response = deepcopy(_fixed_sr_bcrs_response())
        response.levels[0]["axes"][0]["reorder"] = "first_nonzero_col_asc"
        response.capability = {"execution_ready": False}
        issues = _joint_format_issues(response)
        self.assertTrue(any("no executable SM80 backend" in issue for issue in issues))
        self.assertTrue(any("first_nonzero_col_asc" in issue for issue in issues))
        self.assertTrue(any("canonical physicalization rejected" in issue for issue in issues))

    def test_joint_format_retries_unsupported_proposal_before_strategy(self) -> None:
        unsupported = deepcopy(_fixed_sr_bcrs_response())
        unsupported.levels[0]["axes"][0]["reorder"] = "first_nonzero_col_asc"
        unsupported.capability = {"execution_ready": False}
        unsupported.agent_metadata["run_dir"] = "attempt-1"
        accepted = deepcopy(_fixed_sr_bcrs_response())
        accepted.agent_metadata["run_dir"] = "attempt-2"
        for response in (unsupported, accepted):
            response.token_usage = {
                "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0
            }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            feature = root / "matrix.features.v3.json"
            feature.write_text("{}\n", encoding="utf-8")
            args = SimpleNamespace(
                resume=False, format_max_attempts=2, format_retry_delay_sec=0,
                n=512,
            )
            with (
                patch("run_joint_format_strategy_feedback._quality_issues", side_effect=lambda _: []),
                patch("run_joint_format_strategy_feedback._format_config", return_value=SimpleNamespace(model="test", reasoning_effort=None)),
                patch("run_joint_format_strategy_feedback.run_format_design_agent", side_effect=[unsupported, accepted]) as invoke,
                patch("run_joint_format_strategy_feedback.time.sleep"),
            ):
                _, record = _run_format_agent(
                    args=args, dataset="matrix", features={}, feature_path=feature,
                    candidate_dir=root / "format_00", feedback={"outer_iteration": 0},
                )
            self.assertEqual(record["status"], "format_accepted", record)
            self.assertEqual(invoke.call_count, 2)
            first_attempt = (root / "format_00" / "datasets" / "matrix" / "format_attempt_01.json").read_text()
            self.assertIn("first_nonzero_col_asc", first_attempt)
            self.assertIn("campaign_requirement", str(invoke.call_args_list[0].kwargs["quality_feedback"]))

    def test_outer_feedback_omits_raw_ncu_and_keeps_format_level_evidence(self) -> None:
        records = [
            {
                "format_iteration": 0,
                "status": "completed",
                "format_signature": "sig0",
                "format": {"format_name": "candidate0", "params": {"BLK_K": 16}},
                "capability": {"support_level": "integrated_e2e"},
                "strategy_candidate_budget": 6,
                "strategy_candidates_evaluated": 6,
                "strategy_candidates_validated": 6,
                "strategy_candidates_incorrect": 0,
                "strategy_candidates_failed": 0,
                "best_generated_ms": 2.5,
                "best_speedup_vs_cusparse": 1.4,
                "baseline_generated_ms": 2.8,
                "improvement_vs_strategy_baseline": 1.12,
                "best_correctness_passed": True,
                "best_result": {
                    "kernel_visible_format_costs": {"tiles": 123, "row_tiles_p95": 9},
                    "effective_strategy": {"runtime_schedule": {"execution_policy": "row"}},
                },
                "ncu_profile": {"dram_throughput_pct": 88.0},
            }
        ]
        feedback = build_format_feedback(
            dataset="matrix",
            format_iteration=1,
            total_format_iterations=3,
            previous_records=records,
            seed_feedback={"prior_normalized_performance": {"sparsene2_speedup": "1.3"}},
        )
        rendered = str(feedback).lower()
        self.assertNotIn("dram_throughput", rendered)
        self.assertNotIn("ncu_profile", rendered)
        self.assertFalse(feedback["role_boundary"]["format_agent_receives_raw_ncu"])
        self.assertTrue(feedback["selected_format_execution_contract"]["required"])
        history = feedback["previous_outer_iterations"][0]
        self.assertEqual(history["best_kernel_format_level_evidence"]["kernel_visible_format_costs"]["tiles"], 123)
        self.assertEqual(feedback["best_correct_format_so_far"]["format_iteration"], 0)

    def test_best_record_requires_correctness_and_minimum_normal_time(self) -> None:
        records = [
            {"format_iteration": 0, "best_generated_ms": 1.0, "best_correctness_passed": False},
            {"format_iteration": 1, "best_generated_ms": 2.0, "best_correctness_passed": True},
            {"format_iteration": 2, "best_generated_ms": 1.5, "best_correctness_passed": True},
        ]
        self.assertEqual(_best_record(records)["format_iteration"], 2)

    def test_format_signature_is_semantic_not_name_based(self) -> None:
        base = {
            "format_name": "name_a",
            "levels": [{"level": 1, "name": "L", "axes": []}],
            "atomic": {"type": "dense"},
            "params": {"BLK_M": 16, "BLK_K": 8},
        }
        renamed = {**base, "format_name": "name_b"}
        changed = {**base, "params": {"BLK_M": 16, "BLK_K": 16}}
        self.assertEqual(_format_signature(base), _format_signature(renamed))
        self.assertNotEqual(_format_signature(base), _format_signature(changed))

    def test_compact_best_result_excludes_ncu_but_retains_effective_execution(self) -> None:
        entry = {
            "iteration": 4,
            "strategy": {
                "tensor_placement": {"tile_b": 64},
                "pipeline": {"max_wait_prior": 1},
                "compiler": {"max_registers_per_thread": 112},
                "runtime_schedule": {"task_group_tiles": 128},
            },
            "e2e_result": {
                "timing": {"generated_ms": 3.0, "cusparse_ms": 4.5, "speedup_vs_cusparse": 1.5},
                "correctness": {"passed": True},
                "format": {"packing_mode": "column_condensed_dense"},
                "packed": {"tiles": 42, "execution_policy": "strict_lb", "task_groups": 7},
            },
            "ncu_profile": {"registers_per_thread": 112},
        }
        compact = _compact_best_result(entry)
        self.assertEqual(compact["strategy_iteration"], 4)
        self.assertEqual(compact["kernel_visible_format_costs"]["execution_policy"], "strict_lb")
        self.assertNotIn("ncu", str(compact).lower())

    def test_dataset_summary_uses_registry_incumbent_as_performance_floor(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            incumbent_dir = root / "registry" / "datasets" / "matrix"
            incumbent_dir.mkdir(parents=True)
            incumbent_operator = incumbent_dir / "operator.inc"
            incumbent_format = incumbent_dir / "format.json"
            incumbent_operator.write_text("// incumbent\n", encoding="utf-8")
            incumbent_format.write_text("{}\n", encoding="utf-8")
            current_operator = root / "current.inc"
            current_format = root / "current.format.json"
            current_operator.write_text("// current\n", encoding="utf-8")
            current_format.write_text("{}\n", encoding="utf-8")
            incumbent = {
                "schema_version": "sparsekg.e2e_incumbent_registry.v1",
                "dataset": "matrix",
                "candidate_id": "incumbent",
                "source": {"run_id": "previous_run"},
                "selection": {"strategy_iteration": 4},
                "performance": {"generated_ms": 1.0, "speedup_vs_cusparse": 2.0},
                "correctness": {"passed": True},
                "format": {"format_name": "incumbent_format"},
                "strategy": {},
                "artifacts": {"format_sha256": "format-sha"},
                "registry_artifacts": {
                    "operator_path": str(incumbent_operator),
                    "format_path": str(incumbent_format),
                },
            }
            records = [
                {
                    "format_iteration": 0,
                    "status": "completed",
                    "strategy_candidates_evaluated": 6,
                    "strategy_candidates_validated": 6,
                    "strategy_candidates_incorrect": 0,
                    "strategy_candidates_failed": 0,
                    "best_generated_ms": 1.5,
                    "best_speedup_vs_cusparse": 1.5,
                    "best_correctness_passed": True,
                    "best_strategy_iteration": 2,
                    "best_operator_path": str(current_operator),
                    "format_artifact": str(current_format),
                }
            ]
            args = SimpleNamespace(format_iterations=1, strategy_agent_iterations=5)
            summary = _publish_dataset_summary(
                run_dir=root / "run",
                dataset="matrix",
                records=records,
                args=args,
                incumbent=incumbent,
            )
            self.assertEqual(summary["best_source"], "incumbent_registry")
            self.assertEqual(summary["best_source_run"], "previous_run")
            self.assertEqual(summary["best_generated_ms"], 1.0)
            self.assertEqual(summary["best_current_run_generated_ms"], 1.5)
            self.assertEqual(summary["best_current_run_speedup_vs_cusparse"], 1.5)
            self.assertEqual(
                (root / "run" / "datasets" / "matrix" / "best_operator.inc").read_text(),
                "// incumbent\n",
            )

    def test_truncated_format_retry_requests_compact_complete_json(self) -> None:
        response = SimpleNamespace(
            agent_metadata={"source": "fallback", "finish_reason": "length"},
            rationale="Model response was truncated (finish_reason=length)",
            levels=[], atomic={}, params={},
        )
        guidance = _format_retry_guidance(response, ["agent fallback was used"], 1)
        self.assertIn("complete JSON object", guidance["response_budget"])
        self.assertEqual(guidance["previous_dsl"], {})

    def test_joint_format_call_respects_sol_output_budget(self) -> None:
        args = SimpleNamespace(
            format_model="gpt-6.1-sol", format_temperature=0.25,
            format_max_tokens=8192, format_timeout_sec=900,
            format_reasoning_effort=None,
            model_config=Path(__file__).resolve().parents[2] / "llm_models.yaml",
        )
        config = _format_config(args, Path("/tmp/format-agent-test"))
        self.assertEqual(config.max_tokens, 8192)
        self.assertEqual(config.model, "gpt-6.1-sol")


if __name__ == "__main__":
    unittest.main()
