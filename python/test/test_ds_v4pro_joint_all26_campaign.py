"""Offline checks for the all-26 joint campaign wrapper."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from run_ds_v4pro_joint_all26_campaign import DATASETS, probe, summarize, worker_command


class All26CampaignTests(unittest.TestCase):
    def test_dataset_list_is_complete_and_unique(self) -> None:
        self.assertEqual(len(DATASETS), 26)
        self.assertEqual(len(set(DATASETS)), 26)

    def test_worker_uses_historical_dataset_format0_and_short_retry_delay(self) -> None:
        args = SimpleNamespace(
            repo=Path("/repo"), run_dir=Path("/run"), feature_dir=Path("/features"),
            seed_run_dir=Path("/seed"), incumbent_dir=Path("/incumbents"),
            model_config=Path("/repo/llm_models.yaml"), ncu_bin=Path("/ncu"),
        )
        command = worker_command(args, "DD")
        for flag, value in (
            ("--initial-format", "sparsene_historical_best"),
            ("--format-iterations", "3"),
            ("--strategy-agent-iterations", "5"),
            ("--format-retry-delay-sec", "1"),
            ("--n", "512"), ("--warmup", "3"), ("--repeat", "32"),
        ):
            self.assertEqual(command[command.index(flag) + 1], value)
        self.assertIn("--resume", command)
        self.assertNotIn("--abort-on-format-failure", command)

    def test_summary_keeps_missing_workers_visible(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset_dir = root / "workers" / "DD" / "datasets" / "DD"
            dataset_dir.mkdir(parents=True)
            (dataset_dir / "joint_summary.json").write_text(json.dumps({
                "status": "completed_with_gaps",
                "counts": {"format_iterations_recorded": 3, "strategy_candidates_validated": 12},
                "best_current_run_generated_ms": 1.25,
                "best_current_run_speedup_vs_cusparse": 1.5,
                "best_current_run_format_iteration": 1,
                "best_current_run_strategy_iteration": 4,
            }), encoding="utf-8")
            summarize(SimpleNamespace(run_dir=root))
            with (root / "campaign_summary.csv").open(newline="", encoding="utf-8") as handle:
                rows = {row["dataset"]: row for row in csv.DictReader(handle)}
            self.assertEqual(len(rows), 26)
            self.assertEqual(rows["DD"]["status"], "completed_with_gaps")
            self.assertEqual(rows["DD"]["best_current_run_ms"], "1.25")
            self.assertEqual(rows["ASIC_680k"]["status"], "missing")

    def test_probe_uses_configured_stream_route_and_redacts_key(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = SimpleNamespace(
                run_dir=root, model_config=Path(__file__).resolve().parents[2] / "llm_models.yaml"
            )
            response = SimpleNamespace(
                content='{"ok":true}', finish_reason="stop", elapsed_s=0.1,
                usage=SimpleNamespace(to_dict=lambda: {"total_tokens": 2}),
            )
            with patch.dict(os.environ, {"OPENLUX_API_KEY": "probe-test-only-secret"}):
                with patch("sparsene.agent.client.call_openai_compatible", return_value=response) as call:
                    self.assertTrue(probe(args))
            self.assertEqual(call.call_args.kwargs["provider_sort"], "success_rate")
            self.assertTrue(call.call_args.kwargs["stream"])
            self.assertEqual(call.call_args.kwargs["response_format"], {"type": "json_object"})
            records = list((root / "connectivity").glob("deepseek_v4_pro_openlux_probe_*.json"))
            self.assertEqual(len(records), 1)
            record = records[0].read_text()
            self.assertNotIn("probe-test-only-secret", record)


if __name__ == "__main__":
    unittest.main()
