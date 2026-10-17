from __future__ import annotations

from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_joint_format_strategy_feedback import (  # noqa: E402
    _best_record,
    _compact_best_result,
    _format_signature,
    _publish_dataset_summary,
    build_format_feedback,
)


class JointFormatStrategyFeedbackTest(unittest.TestCase):
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
            self.assertEqual(
                (root / "run" / "datasets" / "matrix" / "best_operator.inc").read_text(),
                "// incumbent\n",
            )


if __name__ == "__main__":
    unittest.main()
