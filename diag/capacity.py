#!/usr/bin/env python3
"""Offline, sequential DIII-D oracle-capacity diagnostic.

Run from the workspace: .venv/bin/python -B diag/capacity.py
An unavailable swap reading or missing official mask stops before shot reads.
No network, downloads, mask reconstruction, or modification of source data.
"""
from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True
# Limit numerical libraries in this process; no persistent environment changes.
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import re
import resource
import subprocess
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.linalg import null_space, svd, eigh

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "diag"
STARTER = ROOT / "starter_ref"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(STARTER))
sys.path.insert(0, str(STARTER / "fusion_scoring"))
from greens import coil_matrix, plasma_matrix
from data_fixes import fix_d3d_ip_times
from local_score import build_reference, score_shot, psi_residuals
from common import CONS_SCALARS, N_CONS, N_SCALARS, PSI_SIGNS, SCALARS, SCORING_VERSION
from metrics import Accum, finalize_machine
from lcfs import extract_lcfs

GRIDS = {"V1'": (12, 16), "V3": (24, 32), "V2'": (6, 8)}
VERSIONS = ("truth", "zeros", "V0", *GRIDS, *(v+"_gamma" for v in GRIDS), "PCA20", "PCA50")
BASE_COLUMNS = ["efit_times", "efit_psirz", "efit_grid_R", "efit_grid_Z",
                "efit_q95", "efit_beta_n", "magnetics_time",
                "magnetics_plasma_current", "magnetics_plasma_current_times",
                "coil_input_column", "coil_R", "coil_Z", "coil_width",
                "coil_height", "coil_angle1", "coil_angle2"]


class StopRun(RuntimeError):
    pass


def local_path(path):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f"Path outside workspace: {path}")
    return path


def swap_reading():
    """Fail closed: do not treat an unavailable reading as free memory."""
    if sys.platform.startswith('linux'):
        try:
            raw = Path('/proc/meminfo').read_text()
            match = re.search(r'^SwapFree:\s+(\d+)\s+kB$', raw, re.MULTILINE)
            if not match:
                raise ValueError('SwapFree missing from /proc/meminfo')
            return {"ok": True, "free_mb": int(match[1])/1024, "raw": match[0]}
        except (OSError, ValueError) as e:
            return {"ok": False, "error": str(e), "free_mb": None}
    try:
        p = subprocess.run(["/usr/sbin/sysctl", "vm.swapusage"],
                           capture_output=True, text=True, timeout=5, check=False)
    except (OSError, subprocess.TimeoutExpired) as e:
        return {"ok": False, "error": str(e), "free_mb": None}
    match = re.search(r"free\s*=\s*([\d.]+)([KMGT])", p.stdout)
    if p.returncode or not match:
        return {"ok": False, "error": (p.stderr or p.stdout).strip(), "free_mb": None}
    mb = float(match[1]) * {"K": 1/1024, "M": 1, "G": 1024, "T": 1024**2}[match[2]]
    return {"ok": True, "free_mb": mb, "raw": p.stdout.strip()}


def guard(state, stage, shot, start, budget):
    reading = swap_reading()
    state["swap_checks"].append({"stage": stage, "shot": str(shot), **reading})
    if not reading["ok"]:
        raise StopRun("swap_read_unavailable: " + reading["error"])
    if reading["free_mb"] < 300:
        raise StopRun(f"swap_below_300_mb: {reading['free_mb']} MB")
    if time.monotonic()-start > budget:
        raise StopRun("time_budget_reached")


def arrow_value(scalar):
    """Decode rectangular nested arrays without Python lists of every pixel."""
    # Keep the outer shot list as an Array: constructing a nested ListScalar
    # first can materialize every element before scalar.values is accessed.
    if isinstance(scalar, pa.ChunkedArray):
        scalar = scalar.chunk(0) if scalar.num_chunks == 1 else scalar.combine_chunks()
    if isinstance(scalar, pa.Array):
        if len(scalar) != 1 or scalar.null_count:
            raise ValueError("Expected one nonnull shot column")
        array = scalar.flatten()
    else:
        if not scalar.is_valid:
            raise ValueError("Unexpected null shot column")
        array = scalar.values
    shape = [len(array)]
    while pa.types.is_list(array.type) or pa.types.is_large_list(array.type) or pa.types.is_fixed_size_list(array.type):
        if array.null_count:
            raise ValueError("Unexpected null in requested nested input")
        lengths = np.asarray(array.value_lengths())
        if len(lengths) == 0 or np.any(lengths != lengths[0]):
            raise ValueError("Ragged array is not supported for requested columns")
        shape.append(int(lengths[0]))
        array = array.flatten()
    if pa.types.is_string(array.type) or pa.types.is_large_string(array.type):
        return array.to_pylist()
    if array.null_count:
        raise ValueError("Unexpected null in requested numeric input")
    return array.to_numpy(zero_copy_only=False).reshape(shape)


def read_columns(path, columns):
    pf = pq.ParquetFile(local_path(path))
    if pf.metadata.num_rows != 1:
        raise ValueError(f"Expected one shot per file: {path}")
    table = pf.read(columns=list(dict.fromkeys(columns)), use_threads=False)
    return {name: arrow_value(table[name]) for name in columns}


def load_shot(path):
    keys = read_columns(path, ["coil_input_column"])["coil_input_column"]
    if len(keys) != 19 or set(keys) != ({f"magnetics_F{i}{ab}" for i in range(1, 10) for ab in "AB"} | {"magnetics_ECOILA"}):
        raise ValueError("Unexpected DIII-D coil join keys")
    available = pq.ParquetFile(path).schema_arrow.names
    optional = [c for c in ("efit_lcfs_r", "efit_lcfs_z", "efit_lcfs_n") if c in available]
    row = read_columns(path, BASE_COLUMNS + list(dict.fromkeys(keys)) + optional)
    R, Z = row["efit_grid_R"], row["efit_grid_Z"]
    if len(R) != 65 or len(Z) != 65 or not (np.all(np.diff(R)>0) and np.all(np.diff(Z)>0)):
        raise ValueError("Invalid native EFIT grid")
    if row["efit_psirz"].shape != (len(row["efit_times"]), len(Z), len(R)):
        raise ValueError("Expected psi[time,Z,R]")
    if not np.isfinite(row["efit_psirz"]).all():
        raise ValueError("Nonfinite ground-truth flux")
    return row


def vacuum_and_ip(row, indices, order, split=False, straight=False):
    times = row["efit_times"][indices]
    mt = row["magnetics_time"]
    fixed = fix_d3d_ip_times(row)  # Always apply official per-shot correction.
    if not (np.all(np.diff(mt)>0) and np.all(np.diff(fixed)>0)):
        raise ValueError("Nonmonotone input time axis")
    if times.min() < max(mt[0], fixed[0]) or times.max() > min(mt[-1], fixed[-1]):
        raise ValueError("EFIT times outside current acquisition; refuse silent extrapolation")
    # F currents already include turns. ECOILA: kA -> A, no guessed turns.
    currents = np.column_stack([np.interp(times, mt, row[c])*1000
                                for c in row["coil_input_column"]])
    geometry = dict(row)
    if straight:
        geometry["coil_angle1"] = np.zeros(len(currents[0]))
        geometry["coil_angle2"] = np.zeros(len(currents[0]))
    g = coil_matrix(geometry, row["efit_grid_R"], row["efit_grid_Z"], order)
    vac = currents @ g.T
    ip = np.interp(times, fixed, row["magnetics_plasma_current"])*1000
    corrected = bool(fixed[0] != row["magnetics_plasma_current_times"][0])
    if split:
        e = list(row["coil_input_column"]).index("magnetics_ECOILA")
        ve = currents[:, e, None]*g[:, e]
        return vac-ve, ve, ip, corrected, currents, g
    return vac, ip, corrected


def calibration_sums(vac, truth):
    v = vac - vac.mean(axis=1, keepdims=True)
    y = truth.reshape(len(truth), -1).astype(float)
    y -= y.mean(axis=1, keepdims=True)
    return float(np.sum(v*y)), float(np.sum(v*v))


class ConstrainedFit:
    """Equality-constrained ridge using an orthonormal zero-sum basis.

    Currents are solved in MA, with sum(I)=Ip in native amperes.
    The Green term uses alpha_F; ECOILA calibration never rescales plasma.
    No mask enters this fit. The box may use oracle LCFS geometry.
    """
    def __init__(self, R, Z, nR, nZ, scale, ridge_relative=1e-10, bounds=(1.15, 2.15, -.8, .8)):
        if not np.isfinite(scale) or scale == 0:
            raise ValueError("Nonzero finite physical scale required")
        self.g = scale * 1e6 * plasma_matrix(R, Z, nR, nZ, bounds)
        self.n = self.g.shape[1]
        self.q = null_space(np.ones((1, self.n)))
        self.mean = self.g.mean(axis=0)
        a = (self.g-self.mean) @ self.q
        u, s, vt = svd(a, full_matrices=False, check_finite=True)
        self.ridge = float(ridge_relative*s[0]**2)
        self.solve = ((vt.T * (s/(s*s+self.ridge))) @ u.T)
        self.rank = int(np.sum(s > s[0]*1e-12))

    def fit(self, truth, scaled_vac, ip_A):
        y = np.asarray(truth, float).reshape(len(truth), -1)-scaled_vac
        base = np.repeat((ip_A/1e6/self.n)[:, None], self.n, axis=1)
        r = y-base @ self.g.T
        centered = r-r.mean(axis=1, keepdims=True)
        currents = base + (centered @ self.solve.T) @ self.q.T
        # Correct last floating point rounding; constraint remains exact in A.
        currents[:, -1] += ip_A/1e6-currents.sum(axis=1)
        plasma = currents @ self.g.T
        c = (y-plasma).mean(axis=1)
        pred = scaled_vac+plasma+c[:, None]
        return pred.reshape(np.shape(truth)), currents*1e6, c


def empty_stats():
    return {"psi_sum": 0., "psi_sumsq": 0., "psi_n": 0,
            **{f"{p}_{s}": np.zeros(n) for p, n in (("scal", N_SCALARS), ("cons", N_CONS))
               for s in ("sum", "sumsq", "n")}}


def add_stats(stats, gt, ref):
    y = gt["psi"].astype(float)
    stats["psi_sum"] += float(y.sum())
    stats["psi_sumsq"] += float(np.sum(y*y))
    stats["psi_n"] += y.size
    for prefix, arrays in (("scal", [gt[n] for n in SCALARS]), ("cons", ref[1].T)):
        for j, a in enumerate(arrays):
            a = np.asarray(a)[np.isfinite(a)]
            stats[f"{prefix}_sum"][j] += a.sum()
            stats[f"{prefix}_sumsq"][j] += np.sum(a*a)
            stats[f"{prefix}_n"][j] += a.size


def means_of(stats):
    return (stats["psi_sum"]/stats["psi_n"],
            stats["scal_sum"]/np.maximum(stats["scal_n"], 1),
            stats["cons_sum"]/np.maximum(stats["cons_n"], 1))


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_outputs(state, start):
    state["elapsed_seconds"] = time.monotonic()-start
    state["peak_rss_bytes"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * (1 if sys.platform == "darwin" else 1024)
    state["peak_rss_mib"] = state["peak_rss_bytes"]/1024**2
    HERE.mkdir(exist_ok=True)
    (HERE/"capacity_result.json").write_text(json.dumps(jsonable(state), ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    (HERE/"capacity_report.md").write_text(render_report(state))


def render_report(s):
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
    lines += ["", "内訳の各セルは、負値を0に置き換える前の R² / 失敗フレーム率です。両方の q95,βN 条件で共通です。", "", "|版|"+"|".join(CONS_SCALARS)+"|", "|---|"+"---:|"*N_CONS]
    for v in VERSIONS:
        r = s.get("results", {}).get(v)
        cells = [f"{r['qb_truth']['r2_cons_each'][n]} / {r['qb_truth']['cons_fail_each'][n]}" if r else "未測定" for n in CONS_SCALARS]
        lines.append("|"+v+"|"+"|".join(cells)+"|")
    lines += ["", "校正の切り分け: "+str(s.get("calibration_diagnosis", "未測定")), "", "停止理由: "+str(s.get("stop_reasons", [])), "",
              "前回の実測は capacity_report_run1.md と capacity_result_run1.json に保持します。新しい数値・コイル別校正・PCA学習対象は capacity_result.json と run_* に記録します。", "",
              "合成試験の件数と結果は [verification.json](verification.json) に記録します。合成試験の成功は実データの成功を意味しません。"]
    verification_path = HERE/"verification.json"
    if verification_path.is_file():
        verification = json.loads(verification_path.read_text())
        lines += ["", f"合成データ試験: {verification['tests_run']} 件。合格: {verification['successful']}。独立レビューは行わず、自己点検と実行試験で確認しました。"]
    calibration = s.get("calibration")
    if calibration:
        lines += ["", "## 校正の切り分け", "", "|除外するコイル|α_F|β_E|中心化残差平方和|", "|---|---:|---:|---:|"]
        for name, result in calibration["leave_one_coil_out"].items():
            lines.append(f"|{name}|{result['alpha_F']:.6g}|{result['beta_E']:.6g}|{result['centered_sse']:.6g}|")
        straight = calibration["no_shear"]
        lines += ["", f"傾斜なし: α_F={straight['alpha_F']:.6g}、β_E={straight['beta_E']:.6g}、残差平方和={straight['centered_sse']:.6g}。", "",
                  "コイル別の自由係数と固定倍率候補（±1、±1/(2π)、±0.001、±1000）の残差は JSON に保存します。β_E は α_F と別の直接係数であり、有効巻数の解釈には β_E/α_F を使います。"]
    lines += ["", "修正時の一度だけの補助スクリプト _upgrade_capacity.py、_finish_capacity.py、_finalize_capacity.py は履歴として保持し、再実行を無効にしています。診断や試験からは呼びません。"]
    return "\n".join(lines)+"\n"


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
    active = np.flatnonzero(np.any(m != 0, axis=0))
    aa = a[np.ix_(active, active)]
    if np.linalg.matrix_rank(aa) != len(active):
        raise ValueError("F/ECOILA calibration is not identifiable")
    coef = np.zeros(2)
    coef[active] = np.linalg.solve(aa, b[active])
    return {"alpha_F": float(coef[0]), "beta_E": float(coef[1]),
            "centered_sse": float(yy-2*coef@b+coef@a@coef), "condition_number": float(np.linalg.cond(aa))}


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
            r, z = row["efit_lcfs_r"][j], row["efit_lcfs_z"][j]
            count = int(row["efit_lcfs_n"][j]) if "efit_lcfs_n" in row else len(r)
            if count < 0 or count > len(r) or len(r) != len(z):
                raise ValueError("Invalid stored LCFS point count")
            c = np.column_stack((r[:count], z[:count]))
            c = c[np.isfinite(c).all(axis=1) & (c[:, 0] > 0)]
            c = c if len(c) >= 3 else None
        else:
            c = extract_lcfs(psi, row["efit_grid_R"], row["efit_grid_Z"], "DIII-D", None, None, n_points=512)
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

def prediction_for(version, y, predictions, gamma):
    if version == "truth":
        return y.astype(float)
    if version == "zeros":
        return np.zeros_like(y, dtype=float)
    if version.endswith("_gamma"):
        base = version[:-6]
        return predictions[base+"_p0"]+gamma[base]*predictions[base+"_delta"]
    return predictions[version]


def run(args):
    start = time.monotonic()
    paths = sorted((ROOT/"_data/diii_d_train").glob("*.parquet"))
    state = {"status": "running", "input_shots": len(paths), "scored_shots": 0,
             "scored_frames": 0, "results": {v: None for v in VERSIONS}, "swap_checks": [], "stop_reasons": [],
             "settings": vars(args).copy(), "scoring_version": SCORING_VERSION,
             "shots": [], "stride": args.stride, "stage": "preflight",
             "elapsed_scope": "diagnostic body only; excludes imports, implementation and tests",
             "model": {"equation": "psi=c+alpha_F*F+beta_E*E+alpha_F*gamma*G*I; sum(I)=Ip",
                       "source_counts": {v: nr*nz for v, (nr, nz) in GRIDS.items()},
                       "AF_AF2_coordinate_transform": "unverified cot shear"}}
    try:
        mask_path = local_path(ROOT/args.mask)
        reading = swap_reading()
        state["swap_checks"].append({"stage": "preflight", **reading})
        if not reading["ok"]:
            state["stop_reasons"].append("swap_read_unavailable: " + reading["error"])
        elif reading["free_mb"] < 300:
            state["stop_reasons"].append("swap_below_300_mb")
        if not mask_path.is_file():
            state["stop_reasons"].append("official_mask_missing: " + str(mask_path.relative_to(ROOT)))
        if not paths:
            state["stop_reasons"].append("DIII-D parquet files not found")
        if state["stop_reasons"]:
            state["status"] = "blocked"
            return state
        with np.load(mask_path, allow_pickle=False) as f:
            mask_R, mask_Z = f["grid_R"], f["grid_Z"]
            mask = f["mask_coarse"].astype(bool)
        if mask.shape != (65, 65) or not mask.any():
            raise ValueError("Invalid official envelope mask")
        mask_f = mask.astype(float)
        state["mask_sha256"] = hashlib.sha256(mask_path.read_bytes()).hexdigest()
        state["stage"] = "inventory"
        for path in paths:
            guard(state, "inventory", path.name, start, args.max_seconds)
            times = read_columns(path, ["efit_times"])["efit_times"]
            state["shots"].append({"file": path.name, "frames_total": len(times)})
            del times
        state["input_frames"] = sum(x["frames_total"] for x in state["shots"])
        run_dir = HERE / ("run_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
        run_dir.mkdir()
        state["cache_directory"] = str(run_dir.relative_to(ROOT))
        stats = empty_stats()
        gram = rhs = None
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
                frame_start = time.monotonic()
                prediction, _, _ = fitter.fit(py, pv, pi)
                pr = build_reference(py, R, Z, mask, mask_f)
                pg = {"psi": py, "q95": row["efit_q95"][k], "betaN": row["efit_beta_n"][k]}
                ps = empty_stats(); add_stats(ps, pg, pr)
                score_shot(pg, pr, {"psirz": prediction, "q95": pg["q95"], "betaN": pg["betaN"]}, R, Z, mask, mask_f, 1., means_of(ps))
                per_frame = max(.001, (time.monotonic()-frame_start)/len(k))*len(VERSIONS)*1.5
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
        ablations = {name: solve_calibration(gram, rhs, yy, coil_names, name) for name in coil_names}
        individual = np.linalg.lstsq(gram, rhs, rcond=1e-12)[0]
        fixed = {}
        e = coil_names.index("magnetics_ECOILA")
        f = np.array([i != e for i in range(len(coil_names))], float)
        for alpha in (1., -1., 1/(2*np.pi), -1/(2*np.pi), .001, -.001, 1000., -1000.):
            be = (rhs[e]-alpha*gram[e]@f)/gram[e, e]
            c = alpha*f; c[e] = be
            fixed[str(alpha)] = {"beta_E": float(be), "centered_sse": float(yy-2*c@rhs+c@gram@c)}
        state["calibration"] = {**calibration, "leave_one_coil_out": ablations, "individual_coil_coefficients": dict(zip(coil_names, individual)), "no_shear": solve_calibration(straight_gram, straight_rhs, yy, coil_names), "fixed_alpha_candidates": fixed, "gram": gram, "rhs": rhs}
        influential = sorted(ablations, key=lambda n: abs(ablations[n]["alpha_F"]-scale), reverse=True)[:3]
        no_shear = state["calibration"]["no_shear"]
        shear_gain = (calibration["centered_sse"]-no_shear["centered_sse"])/max(abs(calibration["centered_sse"]), 1e-30)
        unit_candidate = min(fixed, key=lambda k: fixed[k]["centered_sse"])
        state["calibration_diagnosis"] = {
            "most_influential_omitted_coils": influential,
            "no_shear_relative_sse_improvement": shear_gain,
            "best_fixed_alpha_candidate": float(unit_candidate),
            "assessment": ("傾斜変換が疑わしい（傾斜なしで残差が10%以上改善）。" if shear_gain > .1 else "傾斜を除いても残差の10%以上の改善はありません。") +
                " Green係数は独立数値積分で照合し、kA→Aはコード上1回の1000倍です。単位候補の残差とコイル除外結果を併記します。真空校正に省略したプラズマが混ざるため、係数・データ単位・幾何の原因は一意に確定できません。"}

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
        signs = {v: min(PSI_SIGNS, key=lambda sg: totals[v][sg]) for v in VERSIONS}
        state["global_scoring_signs"] = signs
        accs = {(v, cond): Accum() for v in VERSIONS for cond in ("qb_truth", "qb_zero")}
        for (v, _), acc in accs.items():
            acc.psi_sign = signs[v]
        state["stage"] = "official_scoring"
        for si, cache in enumerate(cached):
            guard(state, state["stage"], paths[si].name, start, args.max_seconds)
            with np.load(cache, allow_pickle=False) as f:
                gt = {name: f[name] for name in ("psi", "q95", "betaN")}
                R, Z = f["R"], f["Z"]
                contours = f["contours"]; valid = f["contour_valid"]
                ref = ([c if ok else None for c, ok in zip(contours, valid)], f["cons"], f["cmask"], float(f["rgeo"]))
            with np.load(run_dir/f"shot_{si:02d}_predictions.npz", allow_pickle=False) as pfile:
                for version in VERSIONS:
                    p = prediction_for(version, gt["psi"], pfile, state["gamma"])
                    part = score_shot(gt, ref, {"psirz": p, "q95": gt["q95"], "betaN": gt["betaN"]}, R, Z, mask, mask_f, signs[version], means)
                    accs[version, "qb_truth"].add(part)
                    zero_part = dict(part)
                    zero_part["ss_res_scal"] = np.array([np.sum(gt[name][np.isfinite(gt[name])]**2) for name in SCALARS])
                    accs[version, "qb_zero"].add(zero_part)
                    print(f"score {si+1}/{len(paths)} {version}", flush=True)
            state["scored_shots"] += 1
            state["scored_frames"] += len(gt["psi"])
            # Raw accumulated residuals are retained if a later guard stops.
            # Do not finalize a prefix against whole-corpus reference denominators.
            state["partial_accumulators"] = {f"{v}/{c}": asdict(a) for (v, c), a in accs.items()}
            del gt, ref, contours, p
            gc.collect(); write_outputs(state, start)
        state["consistency_accumulators"] = {v: {"ss_res_cons": accs[v, "qb_truth"].ss_res_cons, "cons_frames": accs[v, "qb_truth"].cons_frames, "cons_fails": accs[v, "qb_truth"].cons_fails} for v in VERSIONS}
        state["results"] = {v: {cond: finalize_machine(accs[v, cond], stats) for cond in ("qb_truth", "qb_zero")} for v in VERSIONS}
        perfect = state["results"]["truth"]["qb_truth"]["S"]
        zero = state["results"]["zeros"]["qb_zero"]["S"]
        state["self_check"] = {"truth_S": perfect, "zeros_S": zero,
                               "passed": abs(perfect-1)<1e-4 and zero<.05}
        state["status"] = "complete" if state["self_check"]["passed"] else "self_check_failed"
        state["stage"] = "finished"
    except StopRun as e:
        state["status"] = "partial" if state["shots"] else "blocked"
        state["stop_reasons"].append(str(e))
    except Exception as e:
        state["status"] = "error"
        state["stop_reasons"].append(f"{type(e).__name__}: {e}")
    finally:
        write_outputs(state, start)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mask", default="starter_ref/fusion_scoring/masks/d3d_envelope.npz")
    parser.add_argument("--stride", type=int, default=0, help="minimum stride; 0: automatic; may increase for V3 runtime")
    parser.add_argument("--max-seconds", type=float, default=1800)
    parser.add_argument("--quadrature-order", type=int, default=6)
    parser.add_argument("--ridge-relative", type=float, default=1e-10)
    args = parser.parse_args()
    if args.stride < 0 or args.max_seconds <= 0 or args.max_seconds > 1800 or args.quadrature_order < 1 or args.ridge_relative <= 0:
        parser.error("Require stride>=0, 0<max-seconds<=1800, order>=1, ridge>0")
    state = run(args)
    print(json.dumps({k: jsonable(state[k]) for k in ("status", "stop_reasons", "elapsed_seconds", "peak_rss_mib")}, ensure_ascii=False))
    return 0 if state["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
