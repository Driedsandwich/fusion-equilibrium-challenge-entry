"""Focused checks for causal inputs, official scoring output and fail-closed guards."""
import contextlib
import io
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import train_eval_v2 as run
from features_v2 import SIGNALS, build_features, feature_names, history, causal_ip_history


class V2Checks(unittest.TestCase):
    def test_window_units_and_empty_windows(self):
        t = np.arange(0., 101., 5.)
        now, windows = history(t, 3*t/1000.+7., np.array([-1., 50., 150.]))
        np.testing.assert_allclose(now, [0., 7.15, 7.3])
        np.testing.assert_allclose([windows[0][1], windows[1][1], windows[2][1], windows[3][1]], [7.135, 3., 7.075, 3.], rtol=1e-6)
        self.assertEqual(windows[0][0], 0.)
        self.assertEqual(windows[1][-1], 0.)

    def test_future_and_targets_are_not_used(self):
        class InputsOnly(dict):
            def __getitem__(self, key):
                if key.startswith('efit_') and key != 'efit_times':
                    raise AssertionError('Target read')
                return super().__getitem__(key)
        times = np.arange(-100., 101., 5.)
        row = InputsOnly(efit_times=np.array([-200., 0., 30.]), magnetics_time=times,
                         magnetics_plasma_current_times=times)
        row.update({f'magnetics_{s}': np.arange(len(times), dtype=float) for s in SIGNALS})
        for system in ('core', 'edge'):
            row[f'thomson_{system}_times'] = times
            for measure in ('Te', 'ne'):
                row[f'thomson_{system}_{measure}'] = np.tile([1., 4., 2.], (len(times), 1))
        expected = build_features(row)
        for key, value in list(row.items()):
            if key.startswith('magnetics_') and not key.endswith(('time', 'times')):
                row[key] = np.where(times > 30., 1e20, value)
            elif key.endswith(('_Te', '_ne')):
                value[times > 30.] = 1e20
        np.testing.assert_array_equal(expected, build_features(row))
        self.assertEqual(expected.shape, (3, len(feature_names())))
        np.testing.assert_array_equal(expected[0], 0.)

    def test_ip_fixer_only_sees_safe_past(self):
        row = dict(magnetics_time=np.array([-3000., 100.]),
                   magnetics_plasma_current_times=np.arange(0., 5000., 100.),
                   magnetics_plasma_current=np.arange(50., dtype=float))
        target = np.array([300.])
        original = causal_ip_history(row, target)
        # These samples could be past under the shifted axis but future under
        # the shipped axis; even the time-origin decision must not inspect them.
        row['magnetics_plasma_current'][4:] = 1e30
        changed = causal_ip_history(row, target)
        for before, after in zip(original, changed):
            np.testing.assert_array_equal(before, after)

    def test_guard_stops_before_data_read(self):
        for swap, free_kb in [({'ok': True, 'free_mb': 299.}, 20*1024**2),
                               ({'ok': True, 'free_mb': 400.}, 12*1024**2-1),
                               ({'ok': False, 'free_mb': None}, 20*1024**2)]:
            with self.subTest(swap=swap, free_kb=free_kb):
                snapshots = []
                disk = SimpleNamespace(returncode=0, stdout=f'Filesystem 1K-blocks Used Available\n/dev/test 1 1 {free_kb}\n')
                with patch.object(sys, 'argv', ['train_eval_v2.py', '--limit-shots', '3']), \
                     patch.object(run, 'swap_reading', return_value=swap), \
                     patch.object(run.subprocess, 'run', return_value=disk), \
                     patch.object(run, 'read_columns') as read, \
                     patch.object(run, 'checkpoint', side_effect=lambda state, *a: (state.update(elapsed_seconds=0.), snapshots.append(dict(state)))), \
                     contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(run.main(), 2)
                    read.assert_not_called()
                self.assertEqual(snapshots[-1]['status'], 'stopped_partial')

    def test_deadline_interrupts_processing(self):
        snapshots = []
        def capture(state, *args):
            state['elapsed_seconds'] = 0.
            snapshots.append(dict(state))
        with patch.object(sys, 'argv', ['train_eval_v2.py', '--limit-shots', '3', '--no-guard', '--max-seconds', '.05']), \
             patch.object(run, 'read_columns', side_effect=lambda *a: time.sleep(.2)), \
             patch.object(run, 'checkpoint', side_effect=capture), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(run.main(), 2)
        self.assertEqual(snapshots[-1]['stop_reason'], 'time_budget_reached')

    def test_guard_bypass_only_allows_three_shots(self):
        with patch.object(sys, 'argv', ['train_eval_v2.py', '--no-guard']), \
             patch.object(run, 'read_columns') as read, \
             contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as result:
                run.main()
            self.assertEqual(result.exception.code, 2)
            read.assert_not_called()


if __name__ == '__main__':
    unittest.main()
