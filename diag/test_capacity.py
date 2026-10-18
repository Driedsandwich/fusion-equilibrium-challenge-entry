"""Synthetic checks only; these do not read DIII-D shots or official masks."""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import json
import unittest
from unittest.mock import patch

import capacity as cap
import numpy as np
import pyarrow as pa
from scipy.integrate import quad
from greens import MU0, loop_flux, rectangle_flux, rectangle_nodes, plasma_matrix, plasma_grid


class PhysicsChecks(unittest.TestCase):
    def test_biot_savart_vector_potential(self):
        # Independent azimuthal integral, rather than another elliptic formula.
        for R, Z, a, b in ((1.3, .2, 1.8, -.4), (.01, 3., 1.2, .5),
                            (2.2, -.2, .8, .1)):
            integrand = lambda phi: np.cos(phi)/np.sqrt(R*R+a*a-2*R*a*np.cos(phi)+(Z-b)**2)
            integral = quad(integrand, 0, 2*np.pi, epsabs=1e-13, epsrel=1e-12)[0]
            expected = MU0/(4*np.pi)*R*a*integral
            self.assertAlmostEqual(float(loop_flux(R, Z, a, b))/expected, 1., places=9)

    def test_axis_limit_reciprocity_and_singularity(self):
        self.assertEqual(float(loop_flux(0, .3, 1.5, 0)), 0.)
        R = 1e-7
        expected = MU0*1.5**2/(2*(1.5**2+.3**2)**1.5)*R**2/2
        self.assertAlmostEqual(float(loop_flux(R, .3, 1.5, 0))/expected, 1., places=9)
        self.assertAlmostEqual(float(loop_flux(1.2, .3, 1.8, -.2)),
                               float(loop_flux(1.8, -.2, 1.2, .3)), places=18)
        self.assertTrue(np.isinf(loop_flux(1.5, 0, 1.5, 0)))

    def test_rectangle_normalization_and_integration(self):
        a, b, weights = rectangle_nodes(1.5, .4, .1, .2, order=8)
        self.assertAlmostEqual(float(weights.sum()), 1.)
        self.assertAlmostEqual(float(a@weights), 1.5)
        self.assertAlmostEqual(float(b@weights), .4)
        point = float(loop_flux(1.1, -.5, 1.5, .4))
        small = float(rectangle_flux(1.1, -.5, 1.5, .4, 1e-6, 1e-6))
        self.assertAlmostEqual(small/point, 1., places=10)
        p6 = rectangle_flux(1.1, -.5, 1.5, .4, .1, .2, 92.4, 0, order=6)
        p10 = rectangle_flux(1.1, -.5, 1.5, .4, .1, .2, 92.4, 0, order=10)
        self.assertAlmostEqual(float(p6/p10), 1., places=10)
        # This checks implemented conventions, not their unverified EFIT provenance.
        straight = rectangle_nodes(1.5, .4, .1, .2, 0, 0)
        ninety = rectangle_nodes(1.5, .4, .1, .2, 90, 90)
        np.testing.assert_allclose(straight[0], ninety[0])
        np.testing.assert_allclose(straight[1], ninety[1])

    def test_constrained_fit_and_shared_sign(self):
        rng = np.random.default_rng(551)
        R, Z = np.linspace(.85, 2.45, 19), np.linspace(-1.4, 1.4, 21)
        g = plasma_matrix(R, Z, 6, 8)
        currents = rng.normal(-2e4, 4e4, (4, 48))
        ip = currents.sum(axis=1)
        vac = rng.normal(0, .01, (4, len(R)*len(Z)))
        scale = -1.2
        true = (scale*(vac+currents@g.T)+np.arange(4)[:, None]*.1).reshape(4, len(Z), len(R))
        fit = cap.ConstrainedFit(R, Z, 6, 8, scale, 1e-15)
        pred, fitted, _ = fit.fit(true, scale*vac, ip)
        np.testing.assert_allclose(fitted.sum(axis=1), ip, atol=1e-8, rtol=0)
        self.assertLess(float(np.max(np.abs(pred-true))), 1e-7)

    def test_global_calibration_with_per_frame_offsets(self):
        rng = np.random.default_rng(19)
        numerator = denominator = 0.
        for frames in (3, 5):
            vacuum = rng.normal(size=(frames, 20))
            true = -1.7*vacuum + rng.normal(size=(frames, 1))
            a, b = cap.calibration_sums(vacuum, true)
            numerator += a; denominator += b
        self.assertAlmostEqual(numerator/denominator, -1.7, places=12)


class RevisedCapacityChecks(unittest.TestCase):
    def test_separate_global_F_and_E_recovery_and_ablation(self):
        rng = np.random.default_rng(405)
        names = ["magnetics_F1A", "magnetics_F2B", "magnetics_ECOILA"]
        gram, rhs, yy = np.zeros((3, 3)), np.zeros(3), 0.
        for frames in (4, 7):
            g, current = rng.normal(size=(40, 3)), rng.normal(size=(frames, 3))
            coef = np.array([-1., -1., 17.])
            truth = (current*coef)@g.T+rng.normal(size=(frames, 1))
            a, b, c = cap.coil_sums(current, g, truth)
            gram += a; rhs += b; yy += c
        result = cap.solve_calibration(gram, rhs, yy, names)
        self.assertAlmostEqual(result["alpha_F"], -1., places=11)
        self.assertAlmostEqual(result["beta_E"], 17., places=11)
        for name in names:
            ablation = cap.solve_calibration(gram, rhs, yy, names, name)
            self.assertGreater(ablation["centered_sse"], result["centered_sse"]+1.)
        self.assertEqual(cap.solve_calibration(gram, rhs, yy, names, names[-1])["beta_E"], 0.)

    def test_degenerate_calibration_refused(self):
        with self.assertRaises(ValueError):
            cap.solve_calibration(np.ones((2, 2)), np.ones(2), 3., ["magnetics_F1A", "magnetics_ECOILA"])

    def test_scale_classification(self):
        for sign in (-1, 1):
            self.assertEqual(cap.classify_scale(sign*1.02), "±1付近")
            self.assertEqual(cap.classify_scale(sign/(2*np.pi)), "±1/(2π)付近")
        self.assertIn("どちらでもない", cap.classify_scale(-.3275))

    def test_box_uses_all_stored_frames_and_padding(self):
        row = {"efit_psirz": np.zeros((3, 2, 2)),
               "efit_lcfs_r": np.array([[1.2, 2.1, 1.7], [1.1, 2.3, 1.6], [1.3, 2.2, 1.6]]),
               "efit_lcfs_z": np.array([[-.8, 0, .8], [-1.3, .1, 1.2], [-1., 0, 1.]])}
        bounds, valid, missing, source = cap.lcfs_bounds(row)
        np.testing.assert_allclose(cap.padded_box(bounds), [1.05, 2.35, -1.35, 1.25])
        self.assertEqual((valid, missing, source), (3, 0, "stored"))
        a, b = plasma_grid(24, 32, cap.padded_box(bounds))
        self.assertEqual(len(a), 768)
        self.assertTrue((b < -1.2).any())

    def test_official_geometry_does_not_receive_mask(self):
        row = {"efit_psirz": np.zeros((2, 3, 3)), "efit_grid_R": np.arange(3), "efit_grid_Z": np.arange(3)}
        contour = np.array([[1., -1.2], [2., 0], [1., 1.3]])
        with patch.object(cap, "extract_lcfs", side_effect=[contour, None]) as extract:
            bounds, valid, missing, _ = cap.lcfs_bounds(row)
        self.assertEqual((valid, missing), (1, 1))
        for call in extract.call_args_list:
            self.assertEqual(call.args[3:], ("DIII-D", None, None))
        np.testing.assert_allclose(bounds, [1., 2., -1.2, 1.3])

    def test_stored_LCFS_point_count_excludes_padding(self):
        row = {"efit_psirz": np.zeros((1, 2, 2)), "efit_lcfs_n": np.array([3]),
               "efit_lcfs_r": np.array([[1.1, 2.2, 1.5, 999.]]),
               "efit_lcfs_z": np.array([[-1.2, 0., 1.3, 999.]])}
        bounds, valid, missing, _ = cap.lcfs_bounds(row)
        np.testing.assert_allclose(bounds, [1.1, 2.2, -1.2, 1.3])
        self.assertEqual((valid, missing), (1, 0))

    def test_free_gamma_recovers_known_plasma_scale(self):
        rng = np.random.default_rng(773)
        R, Z = np.linspace(.8, 2.5, 17), np.linspace(-1.5, 1.5, 19)
        box = (1.05, 2.3, -1.3, 1.25)
        g = plasma_matrix(R, Z, 3, 4, box)
        currents = rng.normal(-50000, 10000, (5, 12))
        ip = currents.sum(axis=1)
        alpha, gamma = -1., 2.7
        vacuum = rng.normal(0, .01, (5, len(R)*len(Z)))
        truth = (vacuum+alpha*gamma*(currents@g.T)+np.arange(5)[:, None]*.1).reshape(5, len(Z), len(R))
        fitter = cap.ConstrainedFit(R, Z, 3, 4, alpha, 1e-15, box)
        p1, i1, _ = fitter.fit(truth, vacuum, ip)
        p0, i0, _ = fitter.fit(truth, vacuum, np.zeros_like(ip))
        delta = p1-p0
        fitted_gamma = np.sum((truth-p0)*delta)/np.sum(delta*delta)
        self.assertAlmostEqual(fitted_gamma, gamma, places=8)
        np.testing.assert_allclose(p0+fitted_gamma*delta, truth, atol=1e-8)
        free_current = i1+(1/fitted_gamma-1)*i0
        np.testing.assert_allclose(free_current.sum(axis=1), ip, atol=1e-8)

    def test_PCA_matches_independent_SVD_and_effective_rank(self):
        rng = np.random.default_rng(882)
        train = rng.normal(size=(80, 65))
        target = rng.normal(size=(7, 65))+3
        result, ranks = cap.pca_project(train, target)
        mean = train.mean(axis=0)
        _, _, vt = np.linalg.svd(train-mean, full_matrices=False)
        for k in (20, 50):
            expected = mean+((target-mean)@vt[:k].T)@vt[:k]
            np.testing.assert_allclose(result[k], expected, atol=1e-10)
            self.assertEqual(ranks[str(k)], k)
        self.assertLess(np.sum((target-result[50])**2), np.sum((target-result[20])**2))

    def test_runtime_increases_explicit_stride(self):
        self.assertGreater(cap.adaptive_stride(2, 2348, 1., 20., 500.), 2)
        with self.assertRaises(cap.StopRun):
            cap.adaptive_stride(2, 100, 1., 400., 300.)

    def test_report_has_all_versions_and_seven_quantities(self):
        report = cap.render_report({"status": "prepared", "results": {}, "stop_reasons": []})
        for name in (*cap.VERSIONS, *cap.CONS_SCALARS, "--stride 2", "直したこと", "α_F", "β_E"):
            self.assertIn(name, report)


class DataAndSafetyChecks(unittest.TestCase):
    def test_arrow_rectangular_arrays(self):
        expected = np.arange(3*5*7, dtype=np.float32).reshape(3, 5, 7)
        scalar = pa.array([expected.tolist()])[0]
        np.testing.assert_array_equal(cap.arrow_value(scalar), expected)
        self.assertEqual(cap.arrow_value(pa.array([["magnetics_F1A", "magnetics_ECOILA"]])[0]),
                         ["magnetics_F1A", "magnetics_ECOILA"])

    def test_swap_units_and_refusal(self):
        from subprocess import CompletedProcess
        with patch.object(cap.subprocess, "run", return_value=CompletedProcess([], 0, "vm.swapusage: total = 10.00G used = 9.50G free = 0.50G", "")):
            self.assertEqual(cap.swap_reading()["free_mb"], 512)
        state = {"swap_checks": []}
        with patch.object(cap, "swap_reading", return_value={"ok": True, "free_mb": 299.9}):
            with self.assertRaises(cap.StopRun):
                cap.guard(state, "test", "synthetic", cap.time.monotonic(), 1800)
        with patch.object(cap, "swap_reading", return_value={"ok": False, "error": "denied", "free_mb": None}):
            args = argparse.Namespace(mask="diag/nonexistent_official_mask.npz", stride=0,
                                      max_seconds=1800., quadrature_order=6, ridge_relative=1e-10)
            with patch.object(cap, "read_columns") as read, patch.object(cap, "write_outputs"):
                result = cap.run(args)
            read.assert_not_called()
            self.assertEqual(result["status"], "blocked")
            self.assertEqual(len(result["stop_reasons"]), 2)

    def test_ip_time_fix_is_applied(self):
        # Simulate a shifted acquisition: shipped current onset is 3 s late.
        times = np.arange(-500., 4500., 10.)
        ip = np.where(times >= 0, 1000., 0.)
        row = {"magnetics_plasma_current_times": times+3000,
               "magnetics_plasma_current": ip, "magnetics_time": times}
        corrected = cap.fix_d3d_ip_times(row)
        np.testing.assert_allclose(corrected, times)


class OfficialScorerSyntheticChecks(unittest.TestCase):
    def test_complete_pipeline_with_synthetic_shots(self):
        # Exercise cache files, pooled means, signs, both scalar conditions,
        # adaptive stride, and the final self-check without reading real shots.
        directory = cap.HERE/"synthetic_checks"
        directory.mkdir(exist_ok=True)
        R, Z = np.linspace(.8, 2.5, 65), np.linspace(-1.5, 1.5, 65)
        rr, zz = np.meshgrid(R, Z)
        mask_path = directory/"mask_synthetic.npz"
        np.savez(mask_path, grid_R=R, grid_Z=Z,
                 mask_coarse=(((rr-1.65)/.7)**2+(zz/1.25)**2 < 1))
        paths = [directory/f"synthetic_{j}.parquet" for j in range(2)]
        rows = {}
        for shot, path in enumerate(paths):
            times = np.arange(-500., 500., 10.)
            et = np.arange(50., 250., 50.)
            row = {"efit_times": et, "efit_grid_R": R, "efit_grid_Z": Z,
                   "magnetics_time": times,
                   "magnetics_plasma_current_times": times+3000,
                   "magnetics_plasma_current": np.where(times>=0, -900.+.1*times, 0.),
                   "coil_input_column": [f"magnetics_F{i}{ab}" for i in range(1, 10) for ab in "AB"]+["magnetics_ECOILA"],
                   "coil_R": np.linspace(.4, .65, 19), "coil_Z": np.linspace(-1.5, 1.5, 19),
                   "coil_width": np.repeat(.01, 19), "coil_height": np.repeat(.02, 19),
                   "coil_angle1": np.zeros(19), "coil_angle2": np.zeros(19),
                   "efit_q95": 3.+.2*shot+np.arange(4)*.1,
                   "efit_beta_n": .8+.1*shot+np.arange(4)*.05}
            for j, name in enumerate(row["coil_input_column"]):
                row[name] = (j+1)*(1+.01*times)+shot
            frames = []
            for j in range(4):
                x, z = rr-(1.60+.012*j+.02*shot), zz-(-.08+.016*j)
                frames.append((x+.08*j*z*z)**2+(z/(1.3+.04*j))**2+.06*j*x**4)
            row["efit_psirz"] = np.stack(frames)
            rows[path] = row
        args = argparse.Namespace(mask=str(mask_path.relative_to(cap.ROOT)), stride=0,
                                  max_seconds=1800., quadrature_order=2, ridge_relative=1e-10)
        def fake_read(path, columns):
            self.assertEqual(columns, ["efit_times"])
            return {"efit_times": rows[path]["efit_times"]}
        with patch.object(Path, "glob", return_value=iter(paths)), \
             patch.object(cap, "HERE", directory), \
             patch.object(cap, "read_columns", side_effect=fake_read), \
             patch.object(cap, "load_shot", side_effect=lambda path: rows[path]), \
             patch.object(cap, "swap_reading", return_value={"ok": True, "free_mb": 1000., "scope": "synthetic"}), \
             patch.object(cap, "write_outputs"):
            state = cap.run(args)
        self.assertEqual(state["status"], "complete", state["stop_reasons"])
        self.assertEqual(state["scored_shots"], 2)
        self.assertEqual(state["scored_frames"], 8)
        self.assertEqual(len(state["swap_checks"]), 9)
        for j, shot in enumerate(state["shots"]):
            self.assertEqual(shot["pca"]["excluded_shot"], paths[j].name)
            self.assertNotIn(paths[j].name, shot["pca"]["training_shots"])
            self.assertEqual(shot["pca"]["training_shots"], [paths[1-j].name])
        self.assertEqual(len(state["gamma"]), 3)
        for version in cap.VERSIONS:
            self.assertEqual(set(state["results"][version]["qb_truth"]["r2_cons_each"]), set(cap.CONS_SCALARS))
        for version in cap.VERSIONS:
            self.assertEqual(state["results"][version]["qb_truth"]["r2_qb"], 1.)
            a, b = (state["results"][version][key] for key in ("qb_truth", "qb_zero"))
            self.assertAlmostEqual(a["S"]-b["S"], .15, places=5)
        self.__class__.pipeline_result = {k: state[k] for k in
                                          ("status", "scored_shots", "scored_frames", "stride", "self_check", "results")}

    def test_synthetic_perfect_and_zeros(self):
        R, Z = np.linspace(.8, 2.5, 65), np.linspace(-1.5, 1.5, 65)
        rr, zz = np.meshgrid(R, Z)
        mask = ((rr-1.65)/.70)**2 + (zz/1.25)**2 < 1
        frames = []
        for j in range(8):
            x, z = rr-(1.60+.012*j), zz-(-.08+.016*j)
            frames.append((x+.08*j*z*z)**2+(z/(1.3+.04*j))**2+.06*j*x**4)
        gt = {"psi": np.stack(frames), "q95": np.linspace(3, 5, 8), "betaN": np.linspace(.8, 2, 8)}
        ref = cap.build_reference(gt["psi"], R, Z, mask, mask.astype(float))
        stats = cap.empty_stats(); cap.add_stats(stats, gt, ref)
        scores = {}
        for name, p in (("truth", gt["psi"]), ("zeros", np.zeros_like(gt["psi"]))):
            pred = {"psirz": p, **{k: gt[k] if name == "truth" else np.zeros(8) for k in cap.SCALARS}}
            part = cap.score_shot(gt, ref, pred, R, Z, mask, mask.astype(float), 1., cap.means_of(stats))
            accumulator = cap.Accum(); accumulator.add(part)
            scores[name] = cap.finalize_machine(accumulator, stats)
        self.assertEqual(scores["truth"]["S"], 1.)
        self.assertLess(scores["zeros"]["S"], .05)
        self.assertGreater(int(stats["cons_n"].min()), 0)
        self.__class__.synthetic_scores = scores


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    evidence = {"scope": "synthetic only; no DIII-D shots or official envelope read",
                "tests_run": result.testsRun, "successful": result.wasSuccessful(),
                "failures": [str(e) for _, e in result.failures],
                "errors": [str(e) for _, e in result.errors],
                "synthetic_official_scores": getattr(OfficialScorerSyntheticChecks, "synthetic_scores", None),
                "synthetic_pipeline": getattr(OfficialScorerSyntheticChecks, "pipeline_result", None)}
    (cap.HERE/"verification.json").write_text(json.dumps(cap.jsonable(evidence), ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    raise SystemExit(0 if result.wasSuccessful() else 1)
