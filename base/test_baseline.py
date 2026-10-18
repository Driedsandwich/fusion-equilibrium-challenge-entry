"""Small regression checks: input isolation and fail-closed resource thresholds."""
import contextlib
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from types import SimpleNamespace

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_eval as run
from features import SIGNALS, build_features
import numpy as np


class BaselineChecks(unittest.TestCase):
    def test_corrected_ip_and_target_isolation(self):
        class InputsOnly(dict):
            def __getitem__(self, key):
                if key.startswith("efit_") and key != "efit_times":
                    raise AssertionError("Feature builder read a target")
                return super().__getitem__(key)
        row = InputsOnly(efit_times=np.array([0., 1., 2.]),
                         magnetics_time=np.arange(-3., 3.),
                         magnetics_plasma_current_times=np.arange(3000., 3006.))
        row.update({f"magnetics_{s}": np.ones(6) * 7 for s in SIGNALS})
        row["magnetics_plasma_current"] = np.array([0., 0., 0., 10., 20., 30.])
        x = build_features(row, True)
        self.assertEqual(x.shape, (3, 33))
        np.testing.assert_array_equal(x[:, 20], [10., 20., 30.])
        np.testing.assert_array_equal(x[:, 21:], 0.)

    def test_guard_stops_before_reading_shot(self):
        # No subprocesses, data reads, or output writes in these threshold tests.
        for swap, disk_kb in [({"ok": True, "free_mb": 299.9}, 20 * 1024**2),
                              ({"ok": True, "free_mb": 500.}, 12 * 1024**2 - 1),
                              ({"ok": False, "free_mb": None}, 20 * 1024**2)]:
            with self.subTest(swap=swap, disk_kb=disk_kb):
                snapshots = []
                def capture(state, args, start):
                    state["elapsed_seconds"] = 0.
                    snapshots.append(dict(state))
                disk = SimpleNamespace(returncode=0, stdout=f"Filesystem 1K-blocks Used Available Capacity Mounted\n/dev/test 1 1 {disk_kb} 1% /\n")
                with patch.object(sys, "argv", ["train_eval.py", "--limit-shots", "3"]), \
                     patch.object(run, "swap_reading", return_value=swap), \
                     patch.object(run.subprocess, "run", return_value=disk), \
                     patch.object(run, "read_columns") as read, \
                     patch.object(run, "checkpoint", side_effect=capture), \
                     contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(run.main(), 2)
                    read.assert_not_called()
                self.assertEqual(snapshots[-1]["status"], "stopped_partial")


if __name__ == "__main__":
    unittest.main()
