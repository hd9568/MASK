from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from sparsene.utils.incumbent_registry import (
    DEFAULT_HARDWARE_SCOPE,
    DEFAULT_PROTOCOL,
    PERFORMANCE_CONTRACT,
    incumbent_as_joint_record,
    load_incumbent,
    refresh_registry_summary,
    update_registry,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class IncumbentRegistryTest(unittest.TestCase):
    def _candidate(self, root: Path, *, candidate_id: str, generated_ms: float) -> dict:
        source = root / "source" / candidate_id
        source.mkdir(parents=True)
        operator = source / "operator.inc"
        operator.write_text(f"// {candidate_id}\n", encoding="utf-8")
        format_path = source / "format.format.json"
        format_path.write_text(
            json.dumps(
                {
                    "format_name": f"format_{candidate_id}",
                    "format_family": "tcf_dense",
                    "atomic": {"type": "dense", "layout": "row_major"},
                    "levels": [],
                    "params": {"BLK_M": 16, "BLK_K": 16},
                }
            ),
            encoding="utf-8",
        )
        manifest = source / "generation_manifest.json"
        manifest.write_text(
            json.dumps({"status": "generated", "operator_sha256": _sha(operator)}),
            encoding="utf-8",
        )
        return {
            "schema_version": "sparsekg.e2e_incumbent_registry.v1",
            "candidate_id": candidate_id,
            "dataset": "matrix",
            "performance_contract": PERFORMANCE_CONTRACT,
            "protocol": dict(DEFAULT_PROTOCOL),
            "hardware": {
                key: DEFAULT_HARDWARE_SCOPE[key]
                for key in ("target", "compute_capability", "sm_count")
            },
            "kernel_provenance": "freshly_generated",
            "source": {
                "kind": "unit_test",
                "run_id": candidate_id,
                "run_dir": str(source),
                "source_summary_path": str(manifest),
                "source_summary_sha256": _sha(manifest),
            },
            "selection": {"format_iteration": 0, "strategy_iteration": 1},
            "performance": {
                "generated_ms": generated_ms,
                "cusparse_ms": 3.0,
                "speedup_vs_cusparse": 3.0 / generated_ms,
                "relative_spread_pct": 0.2,
            },
            "correctness": {
                "passed": True,
                "all_measurements_passed": True,
                "measurement_runs": 3,
            },
            "format": {
                "format_name": f"format_{candidate_id}",
                "format_family": "tcf_dense",
                "atomic_type": "dense",
                "physical": {
                    "logical_blk_m": 16,
                    "physical_blk_m": 16,
                    "blk_k": 16,
                    "packing_mode": "column_condensed_dense",
                    "row_reorder": "none",
                },
                "kernel_visible_costs": {"tiles": 10},
            },
            "strategy": {
                "tensor_placement": {"tile_b": 64},
                "runtime_schedule": {"execution_policy": "row"},
            },
            "artifacts": {
                "operator_path": str(operator),
                "operator_sha256": _sha(operator),
                "format_path": str(format_path),
                "format_sha256": _sha(format_path),
                "generation_manifest_path": str(manifest),
                "generation_manifest_sha256": _sha(manifest),
            },
        }

    def test_only_strictly_faster_compatible_candidate_replaces(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            registry = root / "registry"
            first = self._candidate(root, candidate_id="first", generated_ms=2.0)
            slower = self._candidate(root, candidate_id="slower", generated_ms=2.1)
            faster = self._candidate(root, candidate_id="faster", generated_ms=1.5)

            report = update_registry(registry, [first, slower, faster])
            self.assertEqual(len(report["inserted"]), 1)
            self.assertEqual(len(report["retained"]), 1)
            self.assertEqual(len(report["replaced"]), 1)
            current = load_incumbent(registry, "matrix")
            self.assertIsNotNone(current)
            assert current is not None
            self.assertEqual(current["candidate_id"], "faster")
            self.assertEqual(current["performance"]["generated_ms"], 1.5)
            self.assertEqual(
                (registry / "datasets" / "matrix" / "best_operator.inc").read_text(),
                "// faster\n",
            )
            self.assertEqual(
                len(list((registry / "datasets" / "matrix" / "versions").iterdir())),
                2,
            )
            self.assertEqual(len((registry / "history.jsonl").read_text().splitlines()), 2)
            summary = (registry / "summary.md").read_text(encoding="utf-8")
            self.assertIn("Sparsene 1.0 paper speedup", summary)
            history_before = (registry / "history.jsonl").read_text(encoding="utf-8")
            refresh_registry_summary(registry)
            self.assertEqual(
                (registry / "history.jsonl").read_text(encoding="utf-8"),
                history_before,
            )

            joint = incumbent_as_joint_record(current)
            self.assertTrue(joint["best_correctness_passed"])
            self.assertEqual(joint["best_generated_ms"], 1.5)
            self.assertTrue(Path(joint["best_operator_path"]).is_file())

    def test_protocol_or_correctness_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            registry = root / "registry"
            wrong_n = self._candidate(root, candidate_id="wrong_n", generated_ms=1.0)
            wrong_n["protocol"]["n"] = 256
            incorrect = self._candidate(root, candidate_id="incorrect", generated_ms=0.5)
            incorrect["correctness"]["passed"] = False
            report = update_registry(registry, [wrong_n, incorrect])
            self.assertEqual(len(report["rejected"]), 2)
            self.assertIsNone(load_incumbent(registry, "matrix"))


if __name__ == "__main__":
    unittest.main()
