"""Offline scheduling test: two model workers must not overlap GPU work."""

from __future__ import annotations

import multiprocessing
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_v7_strategy_feedback import _interprocess_gpu_lock  # noqa: E402


def _hold_lock(queue: multiprocessing.Queue, lock_path: str) -> None:
    os.environ["SPARSEKG_GPU_LOCK_FILE"] = lock_path
    with _interprocess_gpu_lock():
        queue.put("enter")
        time.sleep(0.2)
        queue.put("exit")


class InterprocessGpuLockTest(unittest.TestCase):
    def test_two_processes_serialize_gpu_sections(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            lock_path = str(Path(temporary) / "gpu.lock")
            context = multiprocessing.get_context("fork")
            queue = context.Queue()
            processes = [
                context.Process(target=_hold_lock, args=(queue, lock_path))
                for _ in range(2)
            ]
            try:
                for process in processes:
                    process.start()
                events = [queue.get(timeout=5) for _ in range(4)]
                self.assertEqual(events, ["enter", "exit", "enter", "exit"])
            finally:
                for process in processes:
                    process.join(timeout=5)
                    if process.is_alive():
                        process.terminate()
                        process.join(timeout=5)
            self.assertTrue(all(process.exitcode == 0 for process in processes))


if __name__ == "__main__":
    unittest.main()
