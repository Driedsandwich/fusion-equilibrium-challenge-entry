#!/usr/bin/env python3
"""Analytic tests: psi=C*((R-R0)^2+(Z/kappa)^2), q and toroidal pressure."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'mast'), str(ROOT)]
import unittest
import tempfile
import zipfile
from contextlib import nullcontext
import numpy as np
from qb_physics import (MU0, EV_J, vacuum_F, flux_geometry, q_at,
                        volume_average_pressure, beta_n)


def equilibrium(kappa=1., nonuniform=False):
    R0, a, C, F = 5., .5, .7, 8.
    u = np.linspace(-1, 1, 161)
    if nonuniform:
        u = .7*u+.3*u**3
    R, Z = R0+a*u, kappa*a*u
    rr, zz = np.meshgrid(R, Z)
    psi = C*((rr-R0)**2+(zz/kappa)**2)
    return psi, R, Z, R0, a, C, F


class PhysicsTests(unittest.TestCase):
    def check_analytic(self, kappa, nonuniform=False):
        psi, R, Z, r0, a, C, F = equilibrium(kappa, nonuniform)
        g = flux_geometry(psi, R, Z, 'DIII-D')
        self.assertIsNotNone(g['lcfs'])
        expected_q = F*kappa/(2*C*np.sqrt(r0*r0-.95*a*a))
        q = q_at(psi, R, Z, F, machine='DIII-D', geometry=g)
        self.assertAlmostEqual(q/expected_q, 1, delta=.003)
        expected_volume = 2*np.pi**2*r0*kappa*a*a
        self.assertAlmostEqual(g['volume']/expected_volume, 1, delta=.002)
        pn = (np.arange(800)+.5)/800
        sr = r0+a*np.sqrt(pn)
        p0 = 6000.
        ne = np.full_like(sr, 2e19)
        te = p0*(1-pn)/(ne*EV_J)
        pe = volume_average_pressure(psi, R, Z, sr, np.zeros_like(sr), te, ne,
                                     'DIII-D', geometry=g)
        self.assertAlmostEqual(pe/(p0/2), 1, delta=.004)
        print(f'analytic kappa={kappa} nonuniform={nonuniform}: q={q:.9g} expected={expected_q:.9g}; '
              f'<pe>={pe:.9g} expected=3000 Pa', flush=True)

    def test_circular(self):
        self.check_analytic(1.)

    def test_elliptical(self):
        self.check_analytic(1.8)

    def test_nonuniform_physical_grid(self):
        self.check_analytic(1.4, True)

    def test_sign_and_offset_invariance(self):
        p, R, Z, _, _, _, F = equilibrium()
        q = q_at(p, R, Z, F, machine='DIII-D')
        self.assertAlmostEqual(q_at(-p+17, R, Z, -F, machine='MAST'), q, places=8)

    def test_failures_are_nan(self):
        p, R, Z, _, _, _, F = equilibrium()
        for bad in [np.zeros_like(p), np.full_like(p, np.nan), p[:3], np.full_like(p, np.inf)]:
            self.assertTrue(np.isnan(q_at(bad, R, Z, F)))
        self.assertTrue(np.isnan(q_at(p, R[::-1], Z, F)))
        self.assertTrue(np.isnan(beta_n(1, 1, 0, 1)))
        self.assertTrue(np.isnan(volume_average_pressure(p, R, Z, [8, 9], [0, 0],
                                                        [1, 1], [1e19, 1e19], 'DIII-D')))

    def test_units(self):
        self.assertAlmostEqual(float(vacuum_F(-100, 24)), .48)
        self.assertAlmostEqual(beta_n(1000., .5, 2., 1e6), 100*MU0*500.)
        self.assertAlmostEqual(beta_n(1000., .5, 2., -1e6), -100*MU0*500.)
        self.assertEqual(beta_n(0., .5, 2., 1e6), 0.)


class PipelineTests(unittest.TestCase):
    def fixture_dir(self):
        parent = ROOT/'_work/final7/tests'
        parent.mkdir(parents=True, exist_ok=True)
        # Retain fixtures: this task explicitly forbids deleting files.
        return nullcontext(Path(tempfile.mkdtemp(prefix='qb_', dir=parent)))

    def test_d3d_chord_suffix_and_rejection(self):
        from diag.qb_gate_diiid import core_positions
        row = dict(thomson_core_R=np.array([1.94, 1.94]), thomson_core_times=[1, 2],
                   thomson_core_Te=np.ones((2, 2)), thomson_core_ne=np.ones((2, 2)),
                   thomson_chord_name=['TS_core_r+0_1', 'TS_tangential_r+1_0', 'TS_core_r+1_0'],
                   thomson_chord_R=[1.94, 2.1, 1.94], thomson_chord_Z=[.2, 0, .8])
        r, z = core_positions(row)
        np.testing.assert_array_equal(z, [.8, .2])
        row['thomson_chord_name'][-1] = 'TS_core_r+1_1'
        with self.assertRaises(ValueError):
            core_positions(row)

    def test_shared_thomson_time_repair(self):
        from qb_io import repair_profiles
        te = np.arange(10).reshape(5, 2)
        ts, a, b, r, z = repair_profiles([2, np.nan, 1, 1, 3], te, te+100, [1, 2], [0, 0])
        np.testing.assert_array_equal(ts, [1, 2, 3])
        np.testing.assert_array_equal(a, te[[3, 0, 4]])
        np.testing.assert_array_equal(b, a+100)

    def test_pooled_r2_and_mean_substituted_failure(self):
        from diag.qb_gate_diiid import metric_state, metric_add, metric_result
        s = metric_state()
        for y, p in [(1., 1.), (2., np.nan), (4., 3.)]:
            metric_add(s, y, p)
        expected = 1-np.sum((np.array([1., 7/3, 3.])-np.array([1., 2., 4.]))**2)/np.sum((np.array([1., 2., 4.])-7/3)**2)
        self.assertAlmostEqual(metric_result(s)['r2'], expected)

    def test_mast_label_guard_before_projected_read(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        from qb_io import read_mast
        with self.fixture_dir() as d:
            p = Path(d)/'labels.parquet'
            pq.write_table(pa.table({'efit_q95': [[99.]]}), p)
            with self.assertRaisesRegex(ValueError, 'label columns'):
                read_mast(p)

    def test_raw_member_copy(self):
        from qb_assemble import raw_member_copy, store_array
        with self.fixture_dir() as d:
            src, out = Path(d)/'src.npz', Path(d)/'out.npz'
            a = np.arange(500, dtype=np.float16).reshape(20, 25)
            with zipfile.ZipFile(src, 'x', compression=zipfile.ZIP_DEFLATED) as z:
                store_array(z, 'psi', a)
                store_array(z, 'unused', np.zeros(1))
            with zipfile.ZipFile(src) as z, zipfile.ZipFile(out, 'x') as w:
                raw_member_copy(z, w, 'psi.npy')
                store_array(w, 'q95', np.ones(20, np.float16))
            with np.load(out, allow_pickle=False) as z:
                np.testing.assert_array_equal(z['psi'], a)
                self.assertEqual(set(z.files), {'psi', 'q95'})
            with zipfile.ZipFile(src) as s, zipfile.ZipFile(out) as o:
                size = s.infolist()[1].header_offset
                s.fp.seek(0); o.fp.seek(0)
                self.assertEqual(s.fp.read(size), o.fp.read(size))

    def test_guard_stops_when_unavailable_or_low(self):
        import qb_io
        from unittest.mock import patch
        for result in [dict(ok=False, free_mb=None), dict(ok=True, free_mb=299.9)]:
            with patch.object(qb_io, 'swap_reading', return_value=result):
                with self.assertRaises(qb_io.ResourceStop):
                    qb_io.resource_guard()
        with patch.object(qb_io, 'swap_reading', return_value=dict(ok=True, free_mb=300.)):
            qb_io.resource_guard()

    def test_out_of_physical_range_is_filled_like_a_failure(self):
        import qb_io as io
        from qb_generate import finish_chunks, save_chunk
        with self.fixture_dir() as d:
            path = d/'shot_0000.npz'
            save_chunk(path, dict(raw_q95=np.array([3., 1.5, 25., 5.]),
                                  raw_betaN=np.array([1., 9., -0.2, 3.])))
            state = dict(shots=[dict(index=0, chunk=path.name, frames=4, sha256=io.digest(path))])
            self.assertTrue(finish_chunks(d, state, True))
            self.assertEqual(state['global_success_medians'], dict(q95=4., betaN=2.))
            self.assertEqual(state['fill_counts']['q95'], dict(failed=2, shot_median=2, global_median=0))
            with np.load(path, allow_pickle=False) as z:
                self.assertEqual(z['q95'].astype(float).tolist(), [3., 4., 4., 5.])
                self.assertEqual(z['betaN'].astype(float).tolist(), [1., 2., 2., 3.])
                self.assertEqual(z['raw_q95'].tolist(), [3., 1.5, 25., 5.])

    def test_fill_uses_only_successful_estimates(self):
        import qb_io as io
        from qb_generate import finish_chunks, save_chunk
        with self.fixture_dir() as d:
            shots = []
            for i, (q, b) in enumerate([([3., np.nan, 5.], [2., np.nan, 6.]),
                                       ([np.nan, np.nan], [np.nan, np.nan])]):
                path = d/f'shot_{i:04d}.npz'
                save_chunk(path, dict(raw_q95=np.array(q), raw_betaN=np.array(b)))
                shots.append(dict(index=i, chunk=path.name, frames=len(q), sha256=io.digest(path)))
            state = dict(shots=shots)
            self.assertTrue(finish_chunks(d, state, True))
            self.assertEqual(state['global_success_medians'], dict(q95=4., betaN=4.))
            self.assertEqual(state['fill_counts']['betaN'], dict(failed=3, shot_median=1, global_median=2))
            for i in range(2):
                with np.load(d/f'shot_{i:04d}.npz', allow_pickle=False) as z:
                    self.assertEqual(z['q95'].dtype, np.float16)
                    self.assertTrue(np.isfinite(z['q95']).all())
                    self.assertTrue(np.isfinite(z['betaN']).all())
            with np.load(d/'shot_0001.npz', allow_pickle=False) as z:
                np.testing.assert_array_equal(z['betaN'], [4., 4.])
                self.assertTrue(z['failed_betaN'].all())

    def test_no_global_success_does_not_invent_values(self):
        import qb_io as io
        from qb_generate import finish_chunks, save_chunk
        with self.fixture_dir() as d:
            path = d/'shot_0000.npz'
            save_chunk(path, dict(raw_q95=np.array([2.]), raw_betaN=np.array([np.nan])))
            state = dict(shots=[dict(index=0, chunk=path.name, frames=1, sha256=io.digest(path))])
            self.assertFalse(finish_chunks(d, state, True))
            self.assertEqual(state['status'], 'no_successful_estimates')
            self.assertIsNone(state['global_success_medians']['betaN'])

    def test_resume_pending_chunk_replacement(self):
        import qb_io as io
        from qb_generate import recover_pending, save_chunk
        for rename_already_done in [False, True]:
            with self.fixture_dir() as d:
                path = d/'shot_0000.npz'
                save_chunk(path, dict(raw_q95=np.array([1.])))
                old = io.digest(path)
                tmp = path.with_name(path.name+'.partial')
                with tmp.open('wb') as f:
                    np.savez_compressed(f, raw_q95=np.array([2.]))
                new = io.digest(tmp)
                state = dict(shots=[dict(index=0, chunk=path.name, sha256=old)],
                             pending_update=dict(index=0, sha256=new))
                if rename_already_done:
                    tmp.replace(path)
                recover_pending(d, state)
                self.assertEqual(state['shots'][0]['sha256'], new)
                self.assertNotIn('pending_update', state)

    def test_guard_exit2_saves_partial_gate(self):
        import json
        import qb_io
        from unittest.mock import patch
        from diag.qb_gate_diiid import main
        with self.fixture_dir() as d:
            output = d/'gate.json'
            with patch.object(qb_io, 'swap_reading', return_value=dict(ok=True, free_mb=299.)):
                self.assertEqual(main(['--limit-shots', '2', '--output', str(output)]), 2)
            state = json.loads(output.read_text())
            self.assertEqual(state['status'], 'stopped_resource')
            self.assertEqual(state['shots'], [])


if __name__ == '__main__':
    unittest.main(verbosity=2)
