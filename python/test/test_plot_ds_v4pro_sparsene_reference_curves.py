"""Offline checks for historical-reference rebasing and ACC replacement."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from plot_ds_v4pro_sparsene_reference_curves import (
    checked_acc_points,
    earliest_near_best,
)


class HistoricalReferenceCurveTests(unittest.TestCase):
    def test_earliest_noise_equivalent_generated_candidate(self) -> None:
        points = [
            {"format_iteration": 0, "internal_strategy_iteration": 0, "generated_ms": 1.014},
            {"format_iteration": 0, "internal_strategy_iteration": 1, "generated_ms": 1.000},
        ]
        self.assertIs(earliest_near_best(points), points[0])

    def test_new_webbase_format0_requires_mco_and_validated_operator(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            base = run / "datasets/webbase-1M/format_iterations/format_00"
            spec = base / "formats/webbase-1M.format.json"
            spec.parent.mkdir(parents=True)
            spec.write_text(json.dumps({
                "historical_reference": {"variant": "ACC-BIT-TCF"},
                "atomic": {"type": "mco"},
            }), encoding="utf-8")
            iteration = base / "datasets/webbase-1M/strategy_tuning/iteration_00"
            iteration.mkdir(parents=True)
            operator = iteration / "operator.inc"
            operator.write_text("// generated\n", encoding="utf-8")
            sha = hashlib.sha256(operator.read_bytes()).hexdigest()
            manifest = iteration / "generation_manifest.json"
            manifest.write_text(json.dumps({"operator_sha256": sha}), encoding="utf-8")
            summary = iteration / "iteration_summary.json"
            summary.write_text(json.dumps({
                "dataset": "webbase-1M", "iteration": 0, "status": "validated",
                "operator_path": str(operator), "generation_manifest": str(manifest),
                "e2e_result": {
                    "correctness": {"passed": True, "all_measurements_passed": True},
                    "timing": {
                        "contract": "sparsene.kernel_only_post_conversion.v1",
                        "warmup": 3, "repeat": 32, "measurement_runs": 3,
                        "generated_ms": 3.0, "cusparse_ms": 4.0,
                    },
                },
            }), encoding="utf-8")
            points = checked_acc_points(run)
            self.assertEqual(len(points), 1)
            self.assertAlmostEqual(points[0]["speedup_vs_cusparse"], 4 / 3)
            spec.write_text(json.dumps({
                "historical_reference": {"variant": "SR-BCRS-16x8"},
                "atomic": {"type": "dense"},
            }), encoding="utf-8")
            with self.assertRaises(ValueError):
                checked_acc_points(run)


if __name__ == "__main__":
    unittest.main()
