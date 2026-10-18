raise SystemExit("Historical one-off patch script; rerunning is disabled.")
from pathlib import Path
p=Path('diag/greens.py'); s=p.read_text(); a=s.index('def plasma_grid('); s=s[:a]+'''def plasma_grid(nR=12, nZ=16, bounds=(1.15, 2.15, -.8, .8)):
    """Cell centres in an explicit (Rmin,Rmax,Zmin,Zmax) box, in metres."""
    r0, r1, z0, z1 = bounds
    if nR < 1 or nZ < 1 or not (0 < r0 < r1 and z0 < z1):
        raise ValueError("Invalid plasma box or source count")
    r = r0 + (np.arange(nR)+.5)*(r1-r0)/nR
    z = z0 + (np.arange(nZ)+.5)*(z1-z0)/nZ
    a, b = np.meshgrid(r, z)
    return a.ravel(), b.ravel()


def plasma_matrix(R, Z, nR=12, nZ=16, bounds=(1.15, 2.15, -.8, .8)):
    rr, zz = np.meshgrid(R, Z)
    a, b = plasma_grid(nR, nZ, bounds)
    g = loop_flux(rr.ravel()[:, None], zz.ravel()[:, None], a, b)
    if not np.isfinite(g).all():
        raise ValueError("Plasma source coincides with an EFIT grid point")
    return g
'''; p.write_text(s)
p=Path('diag/capacity.py'); s=p.read_text(); s=s.replace('from scipy.linalg import null_space, svd','from scipy.linalg import null_space, svd, eigh'); s=s.replace('from metrics import Accum, finalize_machine','from metrics import Accum, finalize_machine\nfrom lcfs import extract_lcfs'); s=s.replace('VERSIONS = ("truth", "zeros", "V0", "V1", "V2")','GRIDS = {"V1\u2032": (12, 16), "V3": (24, 32), "V2\u2032": (6, 8)}\nVERSIONS = ("truth", "zeros", "V0", *GRIDS, *(v+"_gamma" for v in GRIDS), "PCA20", "PCA50")'); s=s.replace('row = read_columns(path, BASE_COLUMNS + list(dict.fromkeys(keys)))','available = pq.ParquetFile(path).schema_arrow.names\n    optional = [c for c in ("efit_lcfs_r", "efit_lcfs_z") if c in available]\n    row = read_columns(path, BASE_COLUMNS + list(dict.fromkeys(keys)) + optional)'); s=s.replace('def vacuum_and_ip(row, indices, order):','def vacuum_and_ip(row, indices, order, split=False, straight=False):'); s=s.replace('    g = coil_matrix(row, row["efit_grid_R"], row["efit_grid_Z"], order)\n    vac = currents @ g.T','    geometry = dict(row)\n    if straight:\n        geometry["coil_angle1"] = np.zeros(len(currents[0]))\n        geometry["coil_angle2"] = np.zeros(len(currents[0]))\n    g = coil_matrix(geometry, row["efit_grid_R"], row["efit_grid_Z"], order)\n    vac = currents @ g.T'); s=s.replace('    return vac, ip, bool(fixed[0] != row["magnetics_plasma_current_times"][0])','    corrected = bool(fixed[0] != row["magnetics_plasma_current_times"][0])\n    if split:\n        e = list(row["coil_input_column"]).index("magnetics_ECOILA")\n        ve = currents[:, e, None]*g[:, e]\n        return vac-ve, ve, ip, corrected, currents, g\n    return vac, ip, corrected'); s=s.replace('def __init__(self, R, Z, nR, nZ, scale, ridge_relative=1e-10):\n        self.g = scale * 1e6 * plasma_matrix(R, Z, nR, nZ)','def __init__(self, R, Z, nR, nZ, scale, ridge_relative=1e-10, bounds=(1.15, 2.15, -.8, .8)):\n        if not np.isfinite(scale) or scale == 0:\n            raise ValueError("Nonzero finite physical scale required")\n        self.g = scale * 1e6 * plasma_matrix(R, Z, nR, nZ, bounds)'); s=s.replace('    Currents are solved in MA for conditioning. The shared V0 multiplier\n    applies to both vacuum and plasma Green terms, including its sign.\n    No nonnegativity constraint, target LCFS or mask enters this fit.','    Currents are solved in MA, with sum(I)=Ip in native amperes.\n    The Green term uses alpha_F; ECOILA calibration never rescales plasma.\n    No mask enters this fit. The box may use oracle LCFS geometry.'); start=s.index('def render_report(s):'); end=s.index('\ndef run(args):',start)
s=s[:start]+'''def render_report(s):
    lines = ["# 判定①の道具の修正", "", "コードを修正しました。新しい実データの結果は実行後に埋まります。", "",
             "## 直したこと", "",
             "F コイル群の倍率 α_F と ECOILA の係数 β_E を全 shot 共通の2変数最小二乗で分離しました。フレームごとの定数を除いた全画素で当てはめます。",
             "プラズマは α_F G I、ΣI=Ip としました。各格子について、プラズマだけを α_F γ G I とする比較も全 shot 共通の γ で当てはめます。",
             "LCFS（最外殻閉磁気面）は全フレームの境界列を優先し、無ければ公式抽出を使います。採点の包絡マスクを箱の決定や当てはめへ渡しません。各端を5 cm広げます。抽出不能数も記録します。",
             "V1′ は12×16、V3 は24×32、V2′ は6×8です。真値の主成分分析（PCA）の20・50成分への射影は、採点対象の shot を除いた残りの全フレームで基底と平均を作ります。",
             "1コイルずつ除いた校正、コイル別の自由係数、傾斜を無効にした校正、規約候補の固定倍率での誤差も保存します。プラズマを省いた真空校正なので、原因の一意な断定はできません。", "",
             "## 実行と所要時間", "", "```bash", ".venv/bin/python -B diag/test_capacity.py", ".venv/bin/python -B diag/capacity.py --mask starter_ref/fusion_scoring/masks/d3d_envelope.npz --stride 2 --max-seconds 1800", "```", "",
             "実データの実行は Claude が手元で行います。この便では実データを実行しません。スワップ読取不能なら実データの読込前に停止します。",
             "前回の間引き2・184秒・259 MiBに対し、今回は採点版数の増加とPCAを含め、おおよそ6〜15分を見込みます（未実測）。V3 の初回分解と少数フレームの採点を計時し、指定間引きも必要なら自動で増やします。全版は同じフレームで採点します。PCA学習と箱決定は全フレームを使うため、その固定費は間引きでは減りません。30分の上限で未完なら部分結果として停止します。",
             "Green 関数は ψ=R Aφ（Wb/rad）、kA→A は1000倍です。係数は独立した Biot–Savart 数値積分で合成試験します。AF/AF2 の cot による傾斜変換は未検証の仮定です。[公式仕様](../starter_ref/README.md)と[採点器](../starter_ref/fusion_scoring/metrics.py)を参照します。", "",
             f"状態: {s['status']}。採点: {s.get('scored_shots', 0)} shots / {s.get('scored_frames', 0)} フレーム。",
             f"α_F: {s.get('alpha_F', '未測定')}、β_E: {s.get('beta_E', '未測定')}。分類: {s.get('alpha_F_class', '未測定')}。",
             f"自由倍率 γ（格子別）: {s.get('gamma', '未測定')}。",
             f"LCFS範囲・箱（m）: {s.get('geometry', '未測定')}。",
             f"間引き: {s.get('stride', '未決定')}。実行秒数: {s.get('elapsed_seconds', 0):.2f}。ピーク MiB: {s.get('peak_rss_mib', 0):.2f}。", "",
             "R²ψ は磁束の決定係数、D_LCFS は境界距離、Consistency は7量の一致度、S は公式合成点です。内訳は丸め前の残差平方和と失敗数も JSON に保存します。", "",
             "|版|R²ψ|D_LCFS|Consistency|S（q95,βN 真値）|S（q95,βN 0）|", "|---|---:|---:|---:|---:|---:|"]
    for v in VERSIONS:
        r = s.get("results", {}).get(v)
        if r:
            a, b = r["qb_truth"], r["qb_zero"]
            lines.append(f"|{v}|{a['r2_psi']}|{a['dlcfs']}|{a['consistency']}|{a['S']}|{b['S']}|")
        else:
            lines.append(f"|{v}|未測定|未測定|未測定|未測定|未測定|")
    lines += ["", "内訳の各セルは、切り上げ前の R² / 失敗フレーム率です。両方の q95,βN 条件で共通です。", "", "|版|"+"|".join(CONS_SCALARS)+"|", "|---|"+"---:|"*N_CONS]
    for v in VERSIONS:
        r = s.get("results", {}).get(v)
        cells = [f"{r['qb_truth']['r2_cons_each'][n]} / {r['qb_truth']['cons_fail_each'][n]}" if r else "未測定" for n in CONS_SCALARS]
        lines.append("|"+v+"|"+"|".join(cells)+"|")
    lines += ["", "校正の切り分け: "+str(s.get("calibration_diagnosis", "未測定")), "", "停止理由: "+str(s.get("stop_reasons", [])), "",
              "前回の実測は capacity_report_run1.md と capacity_result_run1.json に保持します。新しい数値・コイル別校正・PCA学習対象は capacity_result.json と run_* に記録します。", "",
              "合成試験の件数と結果は [verification.json](verification.json) に記録します。合成試験の成功は実データの成功を意味しません。"]
    return "\\n".join(lines)+"\\n"


def coil_sums(currents, g, truth):
    gc = g-g.mean(axis=0)
    y = np.asarray(truth, float).reshape(len(truth), -1)
    y = y-y.mean(axis=1, keepdims=True)
    return (gc.T@gc)*(currents.T@currents), np.sum((y@gc)*currents, axis=0), float(np.sum(y*y))


def solve_calibration(gram, rhs, yy, names, omit=None):
    m = np.zeros((len(names), 2))
    for j, name in enumerate(names):
        if name != omit:
            m[j, int(name == "magnetics_ECOILA")] = 1
    a, b = m.T@gram@m, m.T@rhs
    if np.linalg.matrix_rank(a) != 2:
        raise ValueError("F/ECOILA calibration is not identifiable")
    coef = np.linalg.solve(a, b)
    return {"alpha_F": float(coef[0]), "beta_E": float(coef[1]),
            "centered_sse": float(yy-2*coef@b+coef@a@coef), "condition_number": float(np.linalg.cond(a))}


def classify_scale(alpha, tolerance=.15):
    for magnitude, label in ((1., "±1付近"), (1/(2*np.pi), "±1/(2π)付近")):
        if abs(abs(alpha)/magnitude-1) <= tolerance:
            return label
    return "どちらでもない（相対許容15%）"


def lcfs_bounds(row):
    points = []
    valid = 0
    stored = "efit_lcfs_r" in row and "efit_lcfs_z" in row
    for j, psi in enumerate(row["efit_psirz"]):
        if stored:
            c = np.column_stack((row["efit_lcfs_r"][j], row["efit_lcfs_z"][j]))
            c = c[np.isfinite(c).all(axis=1) & (c[:, 0] > 0)]
            c = c if len(c) >= 3 else None
        else:
            c = extract_lcfs(psi, row["efit_grid_R"], row["efit_grid_Z"], "d3d", None, None, n_points=512)
        if c is not None and len(c):
            points.append([c[:, 0].min(), c[:, 0].max(), c[:, 1].min(), c[:, 1].max()]); valid += 1
    if not points:
        raise ValueError("No valid ground-truth LCFS to define box")
    points = np.array(points)
    return np.array([points[:, 0].min(), points[:, 1].max(), points[:, 2].min(), points[:, 3].max()]), valid, len(row["efit_psirz"])-valid, "stored" if stored else "official_unmasked"


def padded_box(extents):
    result = np.asarray(extents)+np.array([-.05, .05, -.05, .05])
    if result[0] <= 0:
        raise ValueError("LCFS padding reaches nonpositive R")
    return tuple(result)


def pca_project(train, target, ks=(20, 50)):
    """Exact snapshot PCA; neither centring nor basis observes target shot."""
    train = np.asarray(train, float)
    target = np.asarray(target, float)
    mean = train.mean(axis=0)
    x = train-mean
    gram = x@x.T
    count = min(max(ks), len(train)-1, train.shape[1])
    if count < 1:
        raise ValueError("PCA needs at least two training frames")
    values, vectors = eigh(gram, subset_by_index=(len(train)-count, len(train)-1))
    keep = values > max(values[-1]*1e-12, 0.)
    values, vectors = values[keep][::-1], vectors[:, keep][:, ::-1]
    basis = (vectors.T@x)/np.sqrt(values)[:, None]
    return {k: mean+((target-mean)@basis[:k].T)@basis[:k] for k in ks}, {str(k): min(k, len(basis)) for k in ks}


def adaptive_stride(requested, frames, seconds_per_frame, fixed_seconds, remaining):
    available = remaining*.70-fixed_seconds
    if available <= 0:
        raise StopRun("fixed_cost_exceeds_time_budget")
    return max(1, requested, int(np.ceil(frames*seconds_per_frame/available)))
''' +s[end:]
# Retain safety/scoring skeleton; replace calibration and fitting body below.
a=s.index('             "model": {'); b=s.index('    try:',a); s=s[:a]+'''             "model": {"equation": "psi=c+alpha_F*F+beta_E*E+alpha_F*gamma*G*I; sum(I)=Ip",
                       "source_counts": {v: nr*nz for v, (nr, nz) in GRIDS.items()},
                       "AF_AF2_coordinate_transform": "unverified cot shear"}}
'''+s[b:]
a=s.index('        numerator = denominator = 0.'); b=s.index('        signs = ',a)
s=s[:a]+'''        gram = rhs = None
        yy = 0.
        straight_gram = straight_rhs = None
        extents = []
        cached = []
        state["geometry_shots"] = []
        state["stage"] = "calibration_and_reference"
        for si, path in enumerate(paths):
            guard(state, state["stage"], path.name, start, args.max_seconds)
            row = load_shot(path)
            R, Z = row["efit_grid_R"], row["efit_grid_Z"]
            if not (np.allclose(R, mask_R, atol=1e-12, rtol=0) and np.allclose(Z, mask_Z, atol=1e-12, rtol=0)):
                raise ValueError("Official scoring mask grid differs from EFIT coordinates")
            bounds, valid, missing, source = lcfs_bounds(row)
            extents.append(bounds)
            state["geometry_shots"].append({"file": path.name, "bounds_m": bounds, "valid_frames": valid, "missing_frames": missing, "source": source})
            if si == 0:
                pilot = time.monotonic()
                k = np.arange(min(3, len(row["efit_times"])))
                pv, pi, _ = vacuum_and_ip(row, k, args.quadrature_order)
                py = row["efit_psirz"][k]
                factor_start = time.monotonic()
                fitter = ConstrainedFit(R, Z, 24, 32, 1., args.ridge_relative, padded_box(bounds))
                factor_seconds = time.monotonic()-factor_start
                prediction, _, _ = fitter.fit(py, pv, pi)
                pr = build_reference(py, R, Z, mask, mask_f)
                pg = {"psi": py, "q95": row["efit_q95"][k], "betaN": row["efit_beta_n"][k]}
                ps = empty_stats(); add_stats(ps, pg, pr)
                score_start = time.monotonic()
                score_shot(pg, pr, {"psirz": prediction, "q95": pg["q95"], "betaN": pg["betaN"]}, R, Z, mask, mask_f, 1., means_of(ps))
                per_frame = max(.001, (time.monotonic()-score_start)/len(k))*len(VERSIONS)*1.5
                state["stride"] = adaptive_stride(args.stride, state["input_frames"], per_frame, factor_seconds*2+30, args.max_seconds-(time.monotonic()-start))
                state["pilot"] = {"frames": len(k), "V3_factor_seconds": factor_seconds, "estimated_seconds_per_frame_all_versions": per_frame, "requested_stride": args.stride, "selected_stride": state["stride"]}
                del fitter, prediction, py, pv, pi, pr, pg, ps
            idx = np.arange(0, len(row["efit_times"]), state["stride"])
            gt = {"psi": row["efit_psirz"][idx], "q95": row["efit_q95"][idx], "betaN": row["efit_beta_n"][idx]}
            vf, ve, ip, corrected, currents, g = vacuum_and_ip(row, idx, args.quadrature_order, split=True)
            names = list(row["coil_input_column"])
            if si == 0:
                coil_names = names
                gram = np.zeros((len(names), len(names))); rhs = np.zeros(len(names))
                straight_gram = gram.copy(); straight_rhs = rhs.copy()
            if names != coil_names:
                raise ValueError("Coil order changes across shots")
            cg, ch, cy = coil_sums(currents, g, gt["psi"])
            gram += cg; rhs += ch; yy += cy
            _, _, _, _, ci, sg = vacuum_and_ip(row, idx, args.quadrature_order, split=True, straight=True)
            cg, ch, _ = coil_sums(ci, sg, gt["psi"])
            straight_gram += cg; straight_rhs += ch
            ref = build_reference(gt["psi"], R, Z, mask, mask_f)
            add_stats(stats, gt, ref)
            contours = np.stack([c if c is not None else np.zeros((512, 2)) for c in ref[0]])
            filename = run_dir/f"shot_{si:02d}_reference.npz"
            np.savez(filename, **gt, vacuum_F=vf, vacuum_E=ve, ip_A=ip, R=R, Z=Z,
                     contours=contours, contour_valid=np.array([c is not None for c in ref[0]]),
                     cons=ref[1], cmask=ref[2], rgeo=ref[3], indices=idx)
            np.save(run_dir/f"shot_{si:02d}_pca_truth.npy", row["efit_psirz"].reshape(len(row["efit_times"]), -1))
            cached.append(filename)
            state["shots"][si].update(frames_selected=len(idx), ip_time_corrected=corrected, ip_A_min=float(ip.min()), ip_A_max=float(ip.max()))
            print(f"reference {si+1}/{len(paths)}: frames={len(idx)}, stride={state['stride']}", flush=True)
            del row, gt, vf, ve, ip, currents, g, ci, sg, ref, contours
            gc.collect(); write_outputs(state, start)
        ex = np.array(extents)
        bounds = (ex[:, 0].min(), ex[:, 1].max(), ex[:, 2].min(), ex[:, 3].max())
        box = padded_box(bounds)
        state["geometry"] = {"LCFS_extents_m": bounds, "box_m": box, "padding_m": .05, "frames_valid": sum(x["valid_frames"] for x in state["geometry_shots"]), "frames_missing": sum(x["missing_frames"] for x in state["geometry_shots"])}
        calibration = solve_calibration(gram, rhs, yy, coil_names)
        scale, beta = calibration["alpha_F"], calibration["beta_E"]
        state.update(alpha_F=scale, beta_E=beta, alpha_F_class=classify_scale(scale))
        ablations = {name: solve_calibration(gram, rhs, yy, coil_names, name) for name in coil_names if name != "magnetics_ECOILA"}
        individual = np.linalg.lstsq(gram, rhs, rcond=1e-12)[0]
        fixed = {}
        e = coil_names.index("magnetics_ECOILA")
        f = np.array([i != e for i in range(len(coil_names))], float)
        for alpha in (1., -1., 1/(2*np.pi), -1/(2*np.pi), .001, -.001, 1000., -1000.):
            be = (rhs[e]-alpha*gram[e]@f)/gram[e, e]
            c = alpha*f; c[e] = be
            fixed[str(alpha)] = {"beta_E": float(be), "centered_sse": float(yy-2*c@rhs+c@gram@c)}
        state["calibration"] = {**calibration, "leave_one_F_out": ablations, "individual_coil_coefficients": dict(zip(coil_names, individual)), "no_shear": solve_calibration(straight_gram, straight_rhs, yy, coil_names), "fixed_alpha_candidates": fixed, "gram": gram, "rhs": rhs}
        influential = sorted(ablations, key=lambda n: abs(ablations[n]["alpha_F"]-scale), reverse=True)[:3]
        state["calibration_diagnosis"] = {"most_influential_omitted_coils": influential, "interpretation": "Green coefficient independently checked by synthetic Biot-Savart integration; kA multiplied by 1000 exactly once; compare no_shear and omitted-coil alpha/sse. Plasma omitted from calibration confounds all these tests; causation remains unconfirmed."}
        state["reference_stats"] = stats
        means = means_of(stats)
        state["stage"] = "fit_and_PCA"
        fitters = {v: ConstrainedFit(R, Z, nr, nz, scale, args.ridge_relative, box) for v, (nr, nz) in GRIDS.items()}
        gamma_sums = {v: np.zeros(2) for v in GRIDS}
        for si, cache in enumerate(cached):
            guard(state, state["stage"], paths[si].name, start, args.max_seconds)
            with np.load(cache, allow_pickle=False) as data:
                y, ip = data["psi"], data["ip_A"]
                vacuum = scale*data["vacuum_F"]+beta*data["vacuum_E"]
            flat = y.reshape(len(y), -1)
            c = (flat-vacuum).mean(axis=1)
            predictions = {"V0": (vacuum+c[:, None]).reshape(y.shape)}
            fit_info = {}
            for version, fitter in fitters.items():
                pred, current, offsets = fitter.fit(y, vacuum, ip)
                p0, i0, _ = fitter.fit(y, vacuum, np.zeros_like(ip))
                delta = pred-p0
                gamma_sums[version] += [np.sum((y-p0)*delta), np.sum(delta*delta)]
                predictions.update({version: pred, version+"_p0": p0, version+"_delta": delta, version+"_currents_A": current, version+"_i0": i0, version+"_offsets": offsets})
                fit_info[version] = {"sources": fitter.n, "rank": fitter.rank, "ridge": fitter.ridge, "max_current_constraint_error_A": float(np.max(np.abs(current.sum(axis=1)-ip))), "max_abs_filament_current_A": float(np.max(np.abs(current)))}
            train_ids = [j for j in range(len(cached)) if j != si]
            train = np.concatenate([np.load(run_dir/f"shot_{j:02d}_pca_truth.npy", allow_pickle=False) for j in train_ids]).astype(float)
            projected, ranks = pca_project(train, flat)
            for k, prediction in projected.items():
                predictions[f"PCA{k}"] = prediction.reshape(y.shape)
            state["shots"][si]["pca"] = {"excluded_shot": paths[si].name, "training_shots": [paths[j].name for j in train_ids], "training_frames": len(train), "effective_components": ranks}
            np.savez(run_dir/f"shot_{si:02d}_predictions.npz", **predictions, V0_offsets=c)
            state["shots"][si]["fits"] = fit_info
            del train, projected, predictions, y, flat
            gc.collect(); write_outputs(state, start)
        state["gamma"] = {v: float(a/b) if b > 1e-20 else None for v, (a, b) in gamma_sums.items()}
        if any(g is None or abs(g) < 1e-12 for g in state["gamma"].values()):
            raise ValueError("Global plasma gamma is unidentifiable or zero")
        totals = {v: {sg: 0. for sg in PSI_SIGNS} for v in VERSIONS}
        for si, cache in enumerate(cached):
            with np.load(cache, allow_pickle=False) as data, np.load(run_dir/f"shot_{si:02d}_predictions.npz", allow_pickle=False) as predictions:
                y = data["psi"]
                for v in VERSIONS:
                    p = prediction_for(v, y, predictions, state["gamma"])
                    residuals, _, _ = psi_residuals(y, p, means[0])
                    for sg in PSI_SIGNS:
                        totals[v][sg] += residuals[sg]
''' +s[b:]
s=s.replace('p = gt["psi"].astype(float) if version == "truth" else np.zeros_like(gt["psi"], dtype=float) if version == "zeros" else pfile[version]','p = prediction_for(version, gt["psi"], pfile, state["gamma"])'); s=s.replace('        state["results"] = {v:', '        state["consistency_accumulators"] = {v: {"ss_res_cons": accs[v, "qb_truth"].ss_res_cons, "cons_frames": accs[v, "qb_truth"].cons_frames, "cons_fails": accs[v, "qb_truth"].cons_fails} for v in VERSIONS}\n        state["results"] = {v:')
point=s.index('\ndef run(args):'); s=s[:point]+'''
def prediction_for(version, y, predictions, gamma):
    if version == "truth":
        return y.astype(float)
    if version == "zeros":
        return np.zeros_like(y, dtype=float)
    if version.endswith("_gamma"):
        base = version[:-6]
        return predictions[base+"_p0"]+gamma[base]*predictions[base+"_delta"]
    return predictions[version]

'''+s[point:]; p.write_text(s)
