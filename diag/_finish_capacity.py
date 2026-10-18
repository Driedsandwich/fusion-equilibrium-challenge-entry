raise SystemExit("Historical one-off patch script; rerunning is disabled.")
from pathlib import Path
p=Path('diag/capacity.py'); s=p.read_text().replace('"d3d", None, None, n_points=512','"DIII-D", None, None, n_points=512')
s=s.replace('    if np.linalg.matrix_rank(a) != 2:\n        raise ValueError("F/ECOILA calibration is not identifiable")\n    coef = np.linalg.solve(a, b)','    active = np.flatnonzero(np.any(m != 0, axis=0))\n    aa = a[np.ix_(active, active)]\n    if np.linalg.matrix_rank(aa) != len(active):\n        raise ValueError("F/ECOILA calibration is not identifiable")\n    coef = np.zeros(2)\n    coef[active] = np.linalg.solve(aa, b[active])').replace('float(np.linalg.cond(a))','float(np.linalg.cond(aa))')
s=s.replace('for name in coil_names if name != "magnetics_ECOILA"}', 'for name in coil_names}')
s=s.replace('        state["calibration_diagnosis"] = {"most_influential_omitted_coils": influential, "interpretation": "Green coefficient independently checked by synthetic Biot-Savart integration; kA multiplied by 1000 exactly once; compare no_shear and omitted-coil alpha/sse. Plasma omitted from calibration confounds all these tests; causation remains unconfirmed."}', '''        no_shear = state["calibration"]["no_shear"]
        shear_gain = (calibration["centered_sse"]-no_shear["centered_sse"])/max(abs(calibration["centered_sse"]), 1e-30)
        unit_candidate = min(fixed, key=lambda k: fixed[k]["centered_sse"])
        state["calibration_diagnosis"] = {
            "most_influential_omitted_coils": influential,
            "no_shear_relative_sse_improvement": shear_gain,
            "best_fixed_alpha_candidate": float(unit_candidate),
            "assessment": ("傾斜変換が疑わしい（傾斜なしで残差が10%以上改善）。" if shear_gain > .1 else "傾斜を除いても残差の10%以上の改善はありません。") +
                " Green係数は独立数値積分で照合し、kA→Aはコード上1回の1000倍です。単位候補の残差とコイル除外結果を併記します。真空校正に省略したプラズマが混ざるため、係数・データ単位・幾何の原因は一意に確定できません。"}
''')
s=s.replace('    parser.add_argument("--stride", type=int, default=0, help="0: choose from pilot timing; 1: all frames")','    parser.add_argument("--stride", type=int, default=0, help="minimum stride; 0: automatic; may increase for V3 runtime")')
p.write_text(s)
p=Path('diag/test_capacity.py'); s=p.read_text().replace('plasma_matrix\n','plasma_matrix, plasma_grid\n'); a=s.index('\n\nclass DataAndSafetyChecks')
s=s[:a]+'''

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
''' +s[a:]
s=s.replace('        self.assertEqual(len(state["swap_checks"]), 9)','''        self.assertEqual(len(state["swap_checks"]), 9)
        for j, shot in enumerate(state["shots"]):
            self.assertEqual(shot["pca"]["excluded_shot"], paths[j].name)
            self.assertNotIn(paths[j].name, shot["pca"]["training_shots"])
            self.assertEqual(shot["pca"]["training_shots"], [paths[1-j].name])
        self.assertEqual(len(state["gamma"]), 3)
        for version in cap.VERSIONS:
            self.assertEqual(set(state["results"][version]["qb_truth"]["r2_cons_each"]), set(cap.CONS_SCALARS))''')
p.write_text(s)
