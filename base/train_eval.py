#!/usr/bin/env python3
"""Offline, one-shot-at-a-time PCA/ridge experiment; full run has resource guards."""
from __future__ import annotations

import os
import sys
sys.dont_write_bytecode = True
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"

import argparse
from dataclasses import asdict
import gc
import json
from pathlib import Path
import resource
import subprocess
import time
import traceback

import numpy as np
import pyarrow.parquet as pq
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "diag"))
sys.path.insert(0, str(ROOT / "base"))
from capacity import (StopRun, swap_reading, read_columns, local_path, empty_stats,
                      add_stats, means_of, jsonable, build_reference, score_shot,
                      psi_residuals, Accum, finalize_machine, CONS_SCALARS, SCALARS,
                      SCORING_VERSION)
from features import INPUT_COLUMNS, build_features, feature_names

TARGET_COLUMNS = ["efit_psirz", "efit_q95", "efit_beta_n", "efit_grid_R", "efit_grid_Z"]
VARIANTS = {"a_pca20": (20, False), "b_pca50": (50, False), "c_pca20_thomson": (20, True)}
FULL_COMMAND = ".venv/bin/python -B base/train_eval.py --max-seconds 7200 --output-prefix base/baseline_full"


def peak_mib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024) / 1024**2


def checkpoint(state, args, start):
    state["elapsed_seconds"] = time.monotonic() - start
    state["peak_rss_bytes"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * (1 if sys.platform == "darwin" else 1024)
    state["peak_rss_mib"] = state["peak_rss_bytes"] / 1024**2
    prefix = local_path(ROOT / args.output_prefix)
    prefix.parent.mkdir(exist_ok=True)
    prefix.with_name(prefix.name + "_result.json").write_text(
        json.dumps(jsonable(state), ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    lines = ["# 判定②：公式基線の系統", "",
             "3 shotsの試行は動作確認です。全312 shotsでの門0.85の判定は未実施です。" if args.limit_shots else
             "訓練と保留をshot単位で分け、入力から予測した結果です。", "",
             f"状態: {state['status']}。停止理由: {state.get('stop_reason', 'なし')}。",
             f"訓練 {len(state.get('train_shots', []))} shots / {state.get('train_frames', 0)} フレーム。保留 {len(state.get('holdout_shots', []))} shots / {state.get('holdout_frames', 0)} フレーム。",
             f"採点済み {state.get('scored_shots', 0)} shots / {state.get('scored_frames', 0)} フレーム。採点間引き {args.score_stride}。",
             f"経過 {state['elapsed_seconds']:.2f} 秒。プロセスのピーク記憶 {state['peak_rss_mib']:.2f} MiB。",
             "", "PCA（主成分分析）は訓練shotsの等間隔サンプルだけで平均・50成分を学習します。20成分版はその先頭20成分です。係数のリッジ回帰は全訓練フレームを使い、標準化も訓練だけで当てはめます。リッジの正則化係数は全版1に固定します。",
             "q95・βNは別々に学習します。リッジ回帰とHistGradientBoosting（決定木を順に足す回帰）を比較し、保留の点で設定を調整しません。(a)と(b)の独立回帰は入力が同じなので共有します。各版の学習時間には初回だけ共有回帰の学習を含みます。Thomsonは中心部・周辺部のTe・neの平均・標準偏差・最大値を足します。",
             "公式data_fixesのIp時刻補正を毎回適用します。磁気入力は公式基線と同じ線形補間・範囲外外挿、Thomson要約は範囲外0です。目標は補間しません。包絡マスクは採点だけに渡します。",
             "", f"採点器の版: {SCORING_VERSION}。R²ψは磁束の決定係数、R²qβはq95とβNの平均決定係数、D_LCFSは境界距離、Consistencyは磁束から導く7量の一致度、Sは公式合成点です。",
             "", "|版と独立回帰|R²ψ|R²qβ|D_LCFS|Consistency|S|S≥0.85|", "|---|---:|---:|---:|---:|---:|---|"]
    for name, r in state.get("results", {}).items():
        lines.append(f"|{name}|{r['r2_psi']}|{r['r2_qb']}|{r['dlcfs']}|{r['consistency']}|{r['S']}|{r['S'] >= .85}|")
    lines += ["", "7量の内訳（負値を0に置き換える前の決定係数）:", "",
              "|版|" + "|".join(CONS_SCALARS) + "|", "|---|" + "---:|" * len(CONS_SCALARS)]
    for name, r in state.get("results", {}).items():
        lines.append("|" + name + "|" + "|".join(str(r["r2_cons_each"][c]) for c in CONS_SCALARS) + "|")
    lines += ["", "|磁束の版|学習秒|採点秒|累積ピーク記憶 MiB|", "|---|---:|---:|---:|"]
    for name in VARIANTS:
        lines.append(f"|{name}|{state.get('variant_fit_seconds', {}).get(name)}|{state.get('variant_score_seconds', {}).get(name)}|{state.get('variant_peak_rss_mib', {}).get(name)}|")
    lines += ["", "PCA学習・訓練射影・保留参照の構築は共通の処理です。版別の採点秒は2つの独立回帰の比較を含み、参照の再構築を含みません。同一プロセスのため版別の記憶は、その版の処理終了までの累積最大値です。3 shotsだけの試行では決定係数が負になっています。訓練2 shotsでの数値から全件の性能を判断できません。"]
    lines += ["", "JSONにはshotの識別子、各段階の所要時間、各版の学習・採点時間、q95・βN別の決定係数、7量別の失敗率、資源監視を保存します。途中停止時も完了したshotまでの数値を保存します。部分結果は全保留集合の判定には使えません。", "",
              "Claudeが作業場所のルートで実行する全件命令（全保留フレームを採点）:", "", "```bash", FULL_COMMAND, "```", "",
              "毎回のshot読込前にsysctl vm.swapusageとdf -kの空きを確認します。スワップ空き300 MB未満、ディスク空き12 GiB未満、監視値取得不能、時間上限で途中結果を保存して停止します。--no-guardは--limit-shots 3との併用だけ許します。全件実行では付けません。",
              "", "実装: [features.py](features.py)・[train_eval.py](train_eval.py)。再現用の合成試験: `.venv/bin/python -B base/test_baseline.py`。",
              "", "自己検品と実行確認を行います。独立レビューは依頼の再委譲禁止により行いません。"]
    prefix.with_name(prefix.name + "_report.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit-shots", type=int, default=0)
    parser.add_argument("--no-guard", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--holdout-shots", type=int, default=52)
    parser.add_argument("--pca-frames-per-shot", type=int, default=8)
    parser.add_argument("--score-stride", type=int, default=1)
    parser.add_argument("--max-seconds", type=float, default=7200)
    parser.add_argument("--output-prefix", default="base/baseline")
    args = parser.parse_args()
    if args.no_guard and args.limit_shots != 3:
        parser.error("--no-guard requires --limit-shots 3")
    if args.score_stride < 1 or args.pca_frames_per_shot < 1 or args.max_seconds <= 0:
        parser.error("stride, PCA sample count and time limit must be positive")
    files = sorted((ROOT / "_data/diii_d_train").glob("*.parquet"))
    if args.limit_shots:
        files = files[:args.limit_shots]
    if len(files) < 3:
        parser.error("At least three shots required")
    order = np.random.default_rng(args.seed).permutation(len(files))
    nhold = 1 if args.limit_shots == 3 else args.holdout_shots
    if not 1 <= nhold < len(files):
        parser.error("holdout size must leave at least one training shot")
    hold = [files[i] for i in sorted(order[:nhold])]
    train = [files[i] for i in sorted(order[nhold:])]
    start = time.monotonic()
    state = {"status": "running", "config": vars(args), "scoring_version": SCORING_VERSION,
             "train_shots": [p.name for p in train], "holdout_shots": [p.name for p in hold],
             "resource_checks": [], "timings_seconds": {}, "results": {},
             "train_frames": 0, "holdout_frames": 0, "scored_shots": 0, "scored_frames": 0,
             "features": {"magnetics": feature_names(), "with_thomson": feature_names(True)},
             "full_run_command": FULL_COMMAND}

    def load(path, stage, columns):
        if time.monotonic() - start > args.max_seconds:
            raise StopRun("time_budget_reached")
        if not args.no_guard:
            swap = swap_reading()
            disk = subprocess.run(["df", "-k", str(ROOT)], capture_output=True, text=True,
                                  timeout=5, check=False)
            try:
                free = int(disk.stdout.splitlines()[-1].split()[3]) * 1024
            except (ValueError, IndexError):
                free = None
            state["resource_checks"].append({"stage": stage, "shot": path.name,
                                              "swap": swap, "disk_free_bytes": free})
            if not swap["ok"] or disk.returncode or free is None:
                raise StopRun("resource_read_unavailable")
            if swap["free_mb"] < 300 or free < 12 * 1024**3:
                raise StopRun("resource_below_threshold")
        print(f"{stage}: {path.name}", flush=True)
        available = pq.ParquetFile(path).schema_arrow.names
        required = [c for c in columns if not c.startswith("thomson_") or c in available]
        return read_columns(path, required)

    try:
        stage_start = time.monotonic()
        # Bound sampled flux memory independently of the total training frame count.
        nsample = max(args.pca_frames_per_shot, int(np.ceil(51 / len(train))))
        sample = np.empty((len(train) * nsample, 65 * 65), dtype=np.float32)
        used = 0
        for path in train:
            row = load(path, "pca_sample", ["efit_times", "efit_psirz"])
            y = row["efit_psirz"]
            idx = np.unique(np.linspace(0, len(y) - 1, min(nsample, len(y)), dtype=int))
            sample[used:used + len(idx)] = y[idx].reshape(len(idx), -1)
            used += len(idx)
            del row, y
            gc.collect()
        if used < 51:
            raise ValueError("Need at least 51 training PCA sample frames for the 50-component version")
        pca = PCA(n_components=50, svd_solver="randomized", random_state=args.seed, copy=False)
        pca.fit(sample[:used])
        state["pca_sample_frames"] = used
        state["pca_explained_variance_20_50"] = [float(pca.explained_variance_ratio_[:20].sum()), float(pca.explained_variance_ratio_.sum())]
        del sample
        state["timings_seconds"]["pca"] = time.monotonic() - stage_start
        stage_start = time.monotonic()
        xp, cp, sp = [], [], []
        for path in train:
            row = load(path, "training_projection", INPUT_COLUMNS + ["efit_psirz", "efit_q95", "efit_beta_n"])
            x = build_features(row, True)
            y = row["efit_psirz"].reshape(len(x), -1)
            if not np.isfinite(y).all():
                raise ValueError("Nonfinite training flux")
            xp.append(x)
            cp.append(pca.transform(y).astype(np.float32))
            sp.append(np.column_stack([row["efit_q95"], row["efit_beta_n"]]))
            state["train_frames"] += len(x)
            del row, x, y
            gc.collect()
        x, coeff, scal = np.concatenate(xp), np.concatenate(cp), np.concatenate(sp)
        del xp, cp, sp
        scaler = StandardScaler().fit(x)
        x = scaler.transform(x)
        models, heads = {}, {}
        for name, (n, thomson) in VARIANTS.items():
            st = time.monotonic()
            xx = x if thomson else x[:, :21]
            models[name] = Ridge(alpha=1.).fit(xx, coeff[:, :n])
            if thomson not in heads:
                heads[thomson] = {}
                for method in ("ridge", "hist"):
                    regressors = []
                    for j in range(2):
                        valid = np.isfinite(scal[:, j])
                        reg = Ridge(alpha=1.) if method == "ridge" else HistGradientBoostingRegressor(
                            max_iter=100, max_leaf_nodes=15, l2_regularization=1.,
                            early_stopping=False, random_state=args.seed)
                        regressors.append(reg.fit(xx[valid], scal[valid, j]))
                    heads[thomson][method] = regressors
            state.setdefault("variant_fit_seconds", {})[name] = time.monotonic() - st
            state.setdefault("variant_peak_rss_mib", {})[name] = peak_mib()
        del x, coeff, scal
        gc.collect()
        state["timings_seconds"]["regression_training"] = time.monotonic() - stage_start
        checkpoint(state, args, start)

        def predict(row, name):
            n, thomson = VARIANTS[name]
            xx = scaler.transform(build_features(row, True))
            xx = xx if thomson else xx[:, :21]
            c = models[name].predict(xx)
            psi = (c @ pca.components_[:n] + pca.mean_).reshape(-1, 65, 65)
            return psi, {m: np.column_stack([r.predict(xx) for r in regs])
                         for m, regs in heads[thomson].items()}

        # The mask is loaded only after all fitting is finished.
        with np.load(ROOT / "starter_ref/fusion_scoring/masks/d3d_envelope.npz", allow_pickle=False) as f:
            R, Z, mask = f["grid_R"], f["grid_Z"], f["mask_coarse"].astype(bool)
        mask_f = mask.astype(float)

        def target(row):
            if not (np.array_equal(R, row["efit_grid_R"]) and np.array_equal(Z, row["efit_grid_Z"])):
                raise ValueError("Scoring mask/native grid mismatch")
            idx = slice(None, None, args.score_stride)
            return {"psi": row["efit_psirz"][idx], "q95": row["efit_q95"][idx],
                    "betaN": row["efit_beta_n"][idx]}

        stats = empty_stats()
        signs = {name: {1.: 0., -1.: 0.} for name in VARIANTS}
        stage_start = time.monotonic()
        for path in hold:
            row = load(path, "holdout_reference", INPUT_COLUMNS + TARGET_COLUMNS)
            gt = target(row)
            ref = build_reference(gt["psi"], R, Z, mask, mask_f)
            add_stats(stats, gt, ref)
            state["holdout_frames"] += len(row["efit_times"])
            for name in VARIANTS:
                psi, qb = predict(row, name)
                rr, _, _ = psi_residuals(gt["psi"], psi[::args.score_stride], 0.)
                for sign in signs[name]:
                    signs[name][sign] += rr[sign]
            del row, gt, ref, psi, qb
            gc.collect()
        state["timings_seconds"]["holdout_reference_and_sign"] = time.monotonic() - stage_start
        means = means_of(stats)
        accs = {(name, method): Accum() for name in VARIANTS for method in ("ridge", "hist")}
        for (name, method), acc in accs.items():
            acc.psi_sign = min(signs[name], key=signs[name].get)
        # Prefix stats ensure an interrupted scoring pass never uses whole-fold denominators.
        scored_stats = empty_stats()
        state["variant_score_seconds"] = {n: 0. for n in VARIANTS}
        stage_start = time.monotonic()
        for path in hold:
            row = load(path, "holdout_score", INPUT_COLUMNS + TARGET_COLUMNS)
            gt = target(row)
            ref = build_reference(gt["psi"], R, Z, mask, mask_f)
            for name in VARIANTS:
                st = time.monotonic()
                psi, qb = predict(row, name)
                pred = {"psirz": psi[::args.score_stride],
                        **{s: qb["ridge"][::args.score_stride, j] for j, s in enumerate(SCALARS)}}
                part = score_shot(gt, ref, pred, R, Z, mask, mask_f, accs[name, "ridge"].psi_sign, means)
                accs[name, "ridge"].add(part)
                # LCFS and all seven derived quantities are identical for the two heads.
                other = dict(part)
                residuals = []
                for j, s in enumerate(SCALARS):
                    valid = np.isfinite(gt[s])
                    residuals.append(float(np.sum((gt[s][valid] - qb["hist"][::args.score_stride, j][valid])**2)))
                other["ss_res_scal"] = np.array(residuals)
                accs[name, "hist"].add(other)
                state["variant_score_seconds"][name] += time.monotonic() - st
                state["variant_peak_rss_mib"][name] = peak_mib()
            add_stats(scored_stats, gt, ref)
            state["scored_shots"] += 1
            state["scored_frames"] += len(gt["psi"])
            state["results"] = {f"{n}_{m}": finalize_machine(a, scored_stats) for (n, m), a in accs.items()}
            state["reference_stats"] = scored_stats
            state["accumulators"] = {f"{n}_{m}": asdict(a) for (n, m), a in accs.items()}
            checkpoint(state, args, start)
            del row, gt, ref, psi, qb, pred, part, other
            gc.collect()
        state["timings_seconds"]["scoring"] = time.monotonic() - stage_start
        state["status"] = "complete_smoke" if args.limit_shots == 3 else "complete"
    except StopRun as exc:
        state["status"], state["stop_reason"] = "stopped_partial", str(exc)
    except Exception as exc:
        state["status"], state["stop_reason"] = "error", f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    finally:
        checkpoint(state, args, start)
    print(json.dumps({"status": state["status"], "elapsed_seconds": state["elapsed_seconds"],
                      "S": {n: r["S"] for n, r in state["results"].items()}}, indent=2))
    return 0 if state["status"].startswith("complete") else 2


if __name__ == "__main__":
    raise SystemExit(main())
