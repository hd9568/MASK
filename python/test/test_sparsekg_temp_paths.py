from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sparsene.utils.temp_paths import sparsekg_build_dir, sparsekg_toolchain_env
from run_custom_format_e2e import compile_one
from run_v7_strategy_feedback import _storage_only_resume_fingerprints


class SparsekgTempPathsTest(unittest.TestCase):
    def test_builds_are_stable_and_outside_the_persistent_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "run"
            work_a = run / "format_00" / "iteration_00"
            work_b = run / "format_01" / "iteration_00"
            with patch.dict(os.environ, {"SPARSEKG_TMP_DIR": str(root / "scratch")}):
                first = sparsekg_build_dir(work_a)
                self.assertEqual(first, sparsekg_build_dir(work_a))
                self.assertNotEqual(first, sparsekg_build_dir(work_b))
                self.assertTrue(first.is_relative_to(root / "scratch"))
                self.assertFalse(first.is_relative_to(run))
                env = sparsekg_toolchain_env()
                self.assertEqual(env["TMPDIR"], str(root / "scratch" / "toolchain_tmp"))

    def test_custom_format_compiler_places_executable_in_tmp(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            work = root / "run" / "iteration_00"
            work.mkdir(parents=True)
            operator = work / "operator.inc"
            operator.write_text("// generated operator\n", encoding="utf-8")
            manifest = {
                "dataset": "matrix",
                "work_dir": str(work),
                "operator_path": str(operator),
                "backend_plan": {
                    "row_reorder": None,
                    "packing_mode": "column_condensed_dense",
                    "logical_blk_m": 16,
                    "physical_blk_m": 16,
                    "physical_blk_k": 8,
                    "tile_n": 64,
                },
            }
            with patch.dict(os.environ, {"SPARSEKG_TMP_DIR": str(root / "scratch")}):
                build = sparsekg_build_dir(work)
                build.mkdir(parents=True)
                executable = build / "custom-format-e2e"
                executable.touch()
                with patch("run_custom_format_e2e._run") as run:
                    self.assertEqual(compile_one(root / "run", manifest, jobs=1), executable)
                self.assertEqual(run.call_count, 2)
                configure = run.call_args_list[0].args[0]
                self.assertEqual(configure[configure.index("-B") + 1], str(build))
                self.assertFalse((work / "build").exists())

    def test_only_storage_changes_allow_old_measurements(self) -> None:
        old = {
            "n": 512,
            "repeat": 32,
            "campaign_fingerprint": "old-fingerprint",
            "implementation_files": {
                "python/test/run_v7_strategy_feedback.py": "old-a",
                "python/test/run_custom_format_e2e.py": "old-b",
                "python/sparsene/op_gen/nvir/codegen.py": "same",
            },
        }
        new = json.loads(json.dumps(old))
        new["campaign_fingerprint"] = "new-fingerprint"
        new["implementation_files"]["python/test/run_v7_strategy_feedback.py"] = "new-a"
        new["implementation_files"]["python/test/run_custom_format_e2e.py"] = "new-b"
        self.assertEqual(_storage_only_resume_fingerprints(old, new), {"old-fingerprint"})
        repeated = json.loads(json.dumps(new))
        repeated["resume_compatible_fingerprints"] = ["old-fingerprint"]
        self.assertEqual(
            _storage_only_resume_fingerprints(repeated, repeated), {"old-fingerprint"}
        )
        new["repeat"] = 16
        self.assertEqual(_storage_only_resume_fingerprints(old, new), set())
        new["repeat"] = 32
        new["implementation_files"]["python/sparsene/op_gen/nvir/codegen.py"] = "changed"
        self.assertEqual(_storage_only_resume_fingerprints(old, new), set())


if __name__ == "__main__":
    unittest.main()
