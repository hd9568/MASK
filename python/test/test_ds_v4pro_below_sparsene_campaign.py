"""Offline regression checks for DeepSeek V4 Pro below-baseline campaign bookkeeping."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from run_ds_v4pro_below_sparsene_campaign import BASELINE, DEEPSEEK, frozen_targets, select, summarize


class CampaignBookkeepingTests(unittest.TestCase):
    def test_select_and_freeze_targets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            table = root / "comparison.csv"
            with table.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=("dataset", BASELINE, DEEPSEEK))
                writer.writeheader()
                writer.writerows([
                    {"dataset": "slow", BASELINE: "2", DEEPSEEK: "1"},
                    {"dataset": "fast", BASELINE: "1", DEEPSEEK: "2"},
                ])
            args = SimpleNamespace(run_dir=root / "run", table=table)
            self.assertEqual([row["dataset"] for row in select(table)], ["slow"])
            self.assertEqual([row["dataset"] for row in frozen_targets(args, create=True)], ["slow"])
            table.unlink()
            self.assertEqual([row["dataset"] for row in frozen_targets(args, create=False)], ["slow"])

    def test_summary_uses_current_run_not_cross_run_incumbent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace(run_dir=root)
            dataset_dir = root / "workers" / "slow" / "datasets" / "slow"
            dataset_dir.mkdir(parents=True)
            (dataset_dir / "joint_summary.json").write_text(json.dumps({
                "status": "completed", "best_speedup_vs_cusparse": 9.0,
                "best_current_run_speedup_vs_cusparse": 1.5,
                "best_current_run_format_iteration": 2,
                "best_current_run_strategy_iteration": 5,
            }), encoding="utf-8")
            summarize(args, [{"dataset": "slow", BASELINE: "2", DEEPSEEK: "1"}])
            with (root / "per_dataset_summary.csv").open(newline="", encoding="utf-8") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(float(row["new_best_current_run_speedup"]), 1.5)
            self.assertEqual(float(row["new_vs_sparsene"]), 0.75)
            self.assertEqual(row["best_strategy_iteration"], "5")


if __name__ == "__main__":
    unittest.main()
