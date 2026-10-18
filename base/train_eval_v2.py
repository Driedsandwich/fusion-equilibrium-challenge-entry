#!/usr/bin/env python3
"""Offline nonlinear PCA coefficient regressions; sequential shots, bounded runtime."""
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
import signal
import subprocess
import time
import traceback

import numpy as np
import pyarrow.parquet as pq
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits, threadpool_info

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "base"), str(ROOT / "diag")]
from train_eval import (StopRun, swap_reading, read_columns, local_path, empty_stats,
                        add_stats, means_of, jsonable, build_reference, score_shot,
                        psi_residuals, Accum, finalize_machine, CONS_SCALARS,
                        SCORING_VERSION, peak_mib, TARGET_COLUMNS)
from features_v2 import INPUT_COLUMNS, build_features, feature_names

VARIANTS = ("a_ridge20", "d_hist20", "e_mlp20")
FULL_COMMAND = (".venv/bin/python -B base/train_eval_v2.py --train-frames-per-shot 64 "
                "--hist-iterations 60 --mlp-iterations 60 --score-stride 8 "
                "--max-seconds 510 --output-prefix base/v2_full")


def checkpoint(state, args, start):
    state["elapsed_seconds"] = time.monotonic()-start
    state["peak_rss_mib"] = peak_mib()
    prefix = local_path(ROOT / args.output_prefix)
    prefix.parent.mkdir(exist_ok=True)
    prefix.with_name(prefix.name+"_result.json").write_text(
        json.dumps(jsonable(state), ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    lines = ["# 判定②・2回目：非線形回帰と過去窓の特徴", "",
             "3 shotsの動作確認が完了しました。全312 shotsの成績と門0.85の判定は未実測です。"
             if state["status"] == "complete_smoke" else f"実行状態は {state['status']} です。",
             "", f"停止理由: {state.get('stop_reason', 'なし')}。",
             f"学習 {len(state['train_shots'])} shots・{state['train_frames']} フレーム（選択前 {state['available_train_frames']}）。保留 {len(state['holdout_shots'])} shots。",
             f"採点済み {state['scored_shots']} shots・{state['scored_frames']} フレーム。各shotで {args.score_stride} 枚に1枚を採点します。",
             f"経過 {state['elapsed_seconds']:.2f} 秒。累積ピーク記憶 {state['peak_rss_mib']:.2f} MiB。", "",
             "PCA（主成分分析）は学習shotから等間隔に選んだ磁束だけで20成分を作ります。特徴の標準化とMLPの係数の標準化も学習側だけで行います。保留shotは設定選択や早期停止に使いません。MLPの早期停止は学習側の10%のフレームを使い、同じshot内の相関は残ります。",
             f"全版で同じ {len(feature_names())} 特徴と同じ学習フレームを使います。(a)はリッジ、(d)はPCAの各20成分を別々のHistGradientBoosting（決定木を順に足す回帰）、(e)は隠れ層64・32のMLP（多層ニューラルネット）です。q95とβNは全版共通のHistGradientBoostingを各1本使います。",
             "コイル・ECOILA・bcoil・Ipの最新観測と、直前10 ms・50 msの標本平均・最小二乗の傾き（入力の単位/秒）を使います。窓は[t−窓幅,t]です。空窓は0、1点だけの傾きは0です。未来側との補間・外挿はしません。目標の補間もしません。",
             "Ipの公式data_fixesは全shotの電流値で時刻軸を判別するため、そのまま使うと未来を参照します。今回は配布時刻軸と公式の補正候補軸の両方で現在以前となる入力部分だけを公式関数へ渡します。特徴もこの部分だけで計算します。時刻の曖昧さが大きいshotではIpが古くなり、10 ms・50 msの窓が空になる保守的な扱いです。全shotで判別した従来の補正と同じ結果になる保証はありません。",
             "Thomson（電子温度Te・電子密度neの観測）は各時刻以前の最新プロファイルだけを使います。コアは山の値・平均・山の位置・幅・標準偏差、エッジは平均・最大値・標準偏差です。DIII-Dのコアの高さ座標は未収録のため、位置はチャンネル順を0〜1に正規化します。幅は正の値を重みとした位置の標準偏差で、半値全幅ではありません。欠測や観測開始前は0です。",
             "包絡マスクは全学習終了後に採点だけで読みます。shotは常に1本ずつ読み、配列のキャッシュや模型の複製ファイルは作りません。数値計算とParquetの読込は1スレッドです。",
             "", f"公式採点器の版は {SCORING_VERSION} です。R²ψは磁束の決定係数、R²qβはq95・βNの平均決定係数、D_LCFSは境界距離、Consistencyは磁束から計算する7量の一致度、Sは合成点です。符号は公式の方法と同様に保留全体で1つ選びます。", "",
             "|版|R²ψ|R²qβ|D_LCFS|Consistency|S|", "|---|---:|---:|---:|---:|---:|"]
    for name, r in state["results"].items():
        lines.append("|"+name+"|"+"|".join(str(r[k]) for k in ("r2_psi", "r2_qb", "dlcfs", "consistency", "S"))+"|")
    lines += ["", "|版|R²q95|R²βN|", "|---|---:|---:|"]
    for name, r in state["results"].items():
        lines.append(f"|{name}|{r['r2_each']['q95']}|{r['r2_each']['betaN']}|")
    lines += ["", "境界・7量の失敗率もJSONへ保存します。"]
    lines += ["", "7量の内訳は負値を0に置き換える前の決定係数です。", "",
              "|版|"+"|".join(CONS_SCALARS)+"|", "|---|"+"---:|"*len(CONS_SCALARS)]
    for name, r in state["results"].items():
        lines.append("|"+name+"|"+"|".join(str(r["r2_cons_each"][k]) for k in CONS_SCALARS)+"|")
    lines += ["", "版ごとの学習時間は係数回帰だけです。PCA・特徴作成・スカラー回帰の共通時間はJSONのtimings_secondsに分けています。", "",
              "|版|係数学習秒|共通学習を含む秒|採点秒|", "|---|---:|---:|---:|"]
    common = sum(state["timings_seconds"].get(k, 0.) for k in ("pca", "training_projection", "scalar_fit"))
    for name in VARIANTS:
        fit = state.get("variant_fit_seconds", {}).get(name)
        lines.append(f"|{name}|{fit}|{None if fit is None else fit+common}|{state.get('variant_score_seconds', {}).get(name)}|")
    lines += ["", f"MLPの実行エポック数は {state.get('mlp_epochs')} です。最大反復に達すると未収束の警告が出る場合があります。時間を抑えるため、その設定の予測を評価し、収束済みとは扱いません。",
              "", "3 shotsの成績は学習2 shots・保留1 shotでの動作確認です。負の決定係数もそのまま報告し、全件での性能や前便からの改善とは扱いません。前便とは因果的な入力整列・学習フレーム数・採点間隔も異なるため、模型の効果は今回の3版の間で比較します。", "",
              "3 shots試験の再現命令は `.venv/bin/python -B base/train_eval_v2.py --limit-shots 3 --no-guard --score-stride 8 --max-seconds 120` です。因果性・時間窓・資源停止の検査は `.venv/bin/python -B base/test_v2.py`、旧版の回帰検査は `.venv/bin/python -B base/test_baseline.py` で行います。", "",
              "Claudeがルートで行う全312 shots用の命令（AI担当・この便では実行しません）:", "", "```bash", FULL_COMMAND, "```", "",
              "種42・学習260 shots／保留52 shotsは前便と同じ分割です。学習は各shot最大64フレーム（全版最大16,640フレーム）、PCAは各shot8フレーム、採点は各shot8枚に1枚に抑えます。HGBは係数ごと最大60反復・15葉、スカラーは最大100反復・15葉です。MLPは最大60エポック・改善停止8回です。これは2時間未満の軽量学習を狙う設定です。",
              "前便の全件実測168.77秒を参考に、学習量と採点量を抑えた9分以内を狙う設定です。今回の全件所要時間は未実測なので、完走を保証する値ではありません。開始から510秒で時間停止を発生させ、9分の上限に30秒の保存余裕を残します。時間や資源の停止では終了コード2と途中結果を返します。状態complete・採点52 shotsを確認するまでは全件完了と扱いません。",
              "各shotの読込前と各模型の学習前に既存のswap_readingとdfを使います。スワップ空き300 MB未満、ディスク空き12 GiB未満、監視値取得不能で止めます。--no-guardは--limit-shots 3だけで許可し、本番では付けません。出力先は作業場所内のbase配下に限定します。", "",
              "実装は[features_v2.py](features_v2.py)と[train_eval_v2.py](train_eval_v2.py)です。自己検品はfable-checkを適用し、依頼の再委譲禁止に従って独立レビューは行いません。"]
    prefix.with_name(prefix.name+"_report.md").write_text("\n".join(lines)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit-shots", type=int, default=0)
    parser.add_argument("--no-guard", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--holdout-shots", type=int, default=52)
    parser.add_argument("--pca-frames-per-shot", type=int, default=8)
    parser.add_argument("--train-frames-per-shot", type=int, default=64)
    parser.add_argument("--hist-iterations", type=int, default=60)
    parser.add_argument("--mlp-iterations", type=int, default=60)
    parser.add_argument("--score-stride", type=int, default=8)
    parser.add_argument("--max-seconds", type=float, default=510)
    parser.add_argument("--output-prefix", default="base/v2")
    args = parser.parse_args()
    if args.no_guard and args.limit_shots != 3:
        parser.error("--no-guard requires --limit-shots 3")
    if any(getattr(args, k) < 1 for k in ("pca_frames_per_shot", "train_frames_per_shot", "hist_iterations", "mlp_iterations", "score_stride")) or args.max_seconds <= 0 or args.limit_shots < 0:
        parser.error("Invalid positive count/time limit")
    prefix = local_path(ROOT / args.output_prefix)
    if prefix.parent != ROOT / "base":
        parser.error("Output prefix must be directly in base/")
    files = sorted((ROOT / "_data/diii_d_train").glob("*.parquet"))
    if args.limit_shots:
        files = files[:args.limit_shots]
    nhold = 1 if args.limit_shots == 3 else args.holdout_shots
    if len(files) < 3 or not 1 <= nhold < len(files):
        parser.error("Invalid shot split")
    order = np.random.default_rng(args.seed).permutation(len(files))
    hold = [files[i] for i in sorted(order[:nhold])]
    train = [files[i] for i in sorted(order[nhold:])]
    start = time.monotonic()
    state = dict(status="running", config=vars(args), scoring_version=SCORING_VERSION,
                 train_shots=[p.name for p in train], holdout_shots=[p.name for p in hold],
                 train_frames=0, available_train_frames=0, scored_shots=0, scored_frames=0,
                 resource_checks=[], timings_seconds={}, results={}, features=feature_names(),
                 full_run_command=FULL_COMMAND, variant_fit_seconds={}, variant_score_seconds={},
                 input_policy="past_only_including_official_ip_fix_on_dual_axis_safe_prefix",
                 n_components=20, scalar_model="HistGradientBoostingRegressor")

    def check(stage, shot=""):
        if time.monotonic()-start >= args.max_seconds:
            raise StopRun("time_budget_reached")
        if args.no_guard:
            return
        swap = swap_reading()
        disk = subprocess.run(["df", "-k", str(ROOT)], capture_output=True, text=True, timeout=5, check=False)
        try:
            free = int(disk.stdout.splitlines()[-1].split()[3])*1024
        except (IndexError, ValueError):
            free = None
        state["resource_checks"].append(dict(stage=stage, shot=shot, swap=swap, disk_free_bytes=free))
        if not swap["ok"] or disk.returncode or free is None:
            raise StopRun("resource_read_unavailable")
        if swap["free_mb"] < 300 or free < 12*1024**3:
            raise StopRun("resource_below_threshold")

    def load(path, stage, columns):
        check(stage, path.name)
        print(f"{stage}: {path.name}", flush=True)
        available = pq.ParquetFile(path).schema_arrow.names
        return read_columns(path, [c for c in columns if not c.startswith("thomson_") or c in available])

    def deadline(signum, frame):
        raise StopRun("time_budget_reached")

    previous = signal.signal(signal.SIGALRM, deadline)
    signal.setitimer(signal.ITIMER_REAL, args.max_seconds)
    try:
        with threadpool_limits(limits=1):
            state["threadpools"] = threadpool_info()
            st = time.monotonic()
            nsample = max(args.pca_frames_per_shot, int(np.ceil(21/len(train))))
            sample = np.empty((len(train)*nsample, 65*65), np.float32)
            used = 0
            for path in train:
                row = load(path, "pca_sample", ["efit_times", "efit_psirz"])
                idx = np.unique(np.linspace(0, len(row["efit_times"])-1, nsample, dtype=int))
                sample[used:used+len(idx)] = row["efit_psirz"][idx].reshape(len(idx), -1)
                used += len(idx)
                del row
                gc.collect()
            if used < 20:
                raise ValueError("Insufficient PCA frames")
            pca = PCA(n_components=20, svd_solver="randomized", random_state=args.seed, copy=False).fit(sample[:used])
            state["pca_sample_frames"] = used
            state["pca_explained_variance"] = float(pca.explained_variance_ratio_.sum())
            del sample
            state["timings_seconds"]["pca"] = time.monotonic()-st
            st = time.monotonic()
            xp, cp, sp = [], [], []
            for path in train:
                row = load(path, "training_projection", INPUT_COLUMNS+["efit_psirz", "efit_q95", "efit_beta_n"])
                nt = len(row["efit_times"])
                idx = np.unique(np.linspace(0, nt-1, min(nt, args.train_frames_per_shot), dtype=int))
                xp.append(build_features(row, row["efit_times"][idx]))
                cp.append(pca.transform(row["efit_psirz"][idx].reshape(len(idx), -1)))
                sp.append(np.column_stack((row["efit_q95"][idx], row["efit_beta_n"][idx])))
                state["train_frames"] += len(idx)
                state["available_train_frames"] += nt
                del row
                gc.collect()
            x, c, scalars = np.concatenate(xp), np.concatenate(cp), np.concatenate(sp)
            del xp, cp, sp
            if not np.isfinite(c).all():
                raise ValueError("Nonfinite PCA coefficients")
            scaler = StandardScaler().fit(x)
            x = scaler.transform(x)
            cscale = StandardScaler().fit(c)
            state["timings_seconds"]["training_projection"] = time.monotonic()-st
            st = time.monotonic()
            heads = []
            for j in range(2):
                check("scalar_fit")
                valid = np.isfinite(scalars[:, j])
                heads.append(HistGradientBoostingRegressor(max_iter=100, max_leaf_nodes=15,
                    l2_regularization=1., early_stopping=False, random_state=args.seed).fit(x[valid], scalars[valid, j]))
            state["timings_seconds"]["scalar_fit"] = time.monotonic()-st
            models = {}
            for name in VARIANTS:
                check("fit_"+name)
                st = time.monotonic()
                if name == "a_ridge20":
                    models[name] = Ridge(alpha=1.).fit(x, c)
                elif name == "d_hist20":
                    models[name] = []
                    for j in range(20):
                        check(f"hist_component_{j}")
                        models[name].append(HistGradientBoostingRegressor(max_iter=args.hist_iterations,
                            max_leaf_nodes=15, l2_regularization=1., early_stopping=False,
                            random_state=args.seed).fit(x, c[:, j]))
                else:
                    models[name] = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=args.mlp_iterations,
                        early_stopping=True, validation_fraction=.1, n_iter_no_change=8,
                        batch_size=min(256, max(1, int(len(x)*.9))), alpha=.01, random_state=args.seed).fit(x, cscale.transform(c))
                    state["mlp_epochs"] = int(models[name].n_iter_)
                state["variant_fit_seconds"][name] = time.monotonic()-st
                checkpoint(state, args, start)
            del x, c, scalars
            gc.collect()

            def predict(row, name):
                xx = scaler.transform(build_features(row, row["efit_times"][::args.score_stride]))
                model = models[name]
                if name == "d_hist20":
                    coeff = np.column_stack([m.predict(xx) for m in model])
                else:
                    coeff = model.predict(xx)
                    if name == "e_mlp20":
                        coeff = cscale.inverse_transform(coeff)
                psi = (coeff @ pca.components_+pca.mean_).reshape(-1, 65, 65)
                return psi, np.column_stack([h.predict(xx) for h in heads])

            # Mask enters only official scoring, after fitting has finished.
            with np.load(ROOT / "starter_ref/fusion_scoring/masks/d3d_envelope.npz", allow_pickle=False) as f:
                R, Z, mask = f["grid_R"], f["grid_Z"], f["mask_coarse"].astype(bool)
            mask_f = mask.astype(float)

            def target(row):
                if not (np.array_equal(R, row["efit_grid_R"]) and np.array_equal(Z, row["efit_grid_Z"])):
                    raise ValueError("Native grid/mask mismatch")
                return dict(psi=row["efit_psirz"][::args.score_stride], q95=row["efit_q95"][::args.score_stride], betaN=row["efit_beta_n"][::args.score_stride])

            stats = empty_stats()
            signs = {n: {1.: 0., -1.: 0.} for n in VARIANTS}
            st = time.monotonic()
            for path in hold:
                row = load(path, "holdout_reference", INPUT_COLUMNS+TARGET_COLUMNS)
                gt = target(row)
                ref = build_reference(gt["psi"], R, Z, mask, mask_f)
                add_stats(stats, gt, ref)
                for name in VARIANTS:
                    psi, _ = predict(row, name)
                    rr, _, _ = psi_residuals(gt["psi"], psi, 0.)
                    for sign in signs[name]:
                        signs[name][sign] += rr[sign]
                del row, gt, ref, psi
                gc.collect()
            state["timings_seconds"]["holdout_reference_and_sign"] = time.monotonic()-st
            means = means_of(stats)
            accs = {n: Accum() for n in VARIANTS}
            for name, acc in accs.items():
                acc.psi_sign = min(signs[name], key=signs[name].get)
            scored_stats = empty_stats()
            state["variant_score_seconds"] = {n: 0. for n in VARIANTS}
            st = time.monotonic()
            for path in hold:
                row = load(path, "holdout_score", INPUT_COLUMNS+TARGET_COLUMNS)
                gt = target(row)
                ref = build_reference(gt["psi"], R, Z, mask, mask_f)
                # Commit statistics only after all variants finish this shot.
                parts = {}
                for name in VARIANTS:
                    check("score_"+name, path.name)
                    ts = time.monotonic()
                    psi, qb = predict(row, name)
                    pred = dict(psirz=psi, q95=qb[:, 0], betaN=qb[:, 1])
                    parts[name] = score_shot(gt, ref, pred, R, Z, mask, mask_f, accs[name].psi_sign, means)
                    state["variant_score_seconds"][name] += time.monotonic()-ts
                for name in VARIANTS:
                    accs[name].add(parts[name])
                add_stats(scored_stats, gt, ref)
                state["scored_shots"] += 1
                state["scored_frames"] += len(gt["psi"])
                state["results"] = {n: finalize_machine(a, scored_stats) for n, a in accs.items()}
                state["reference_stats"] = scored_stats
                state["accumulators"] = {n: asdict(a) for n, a in accs.items()}
                checkpoint(state, args, start)
                del row, gt, ref, psi, qb, pred, parts
                gc.collect()
            state["timings_seconds"]["scoring"] = time.monotonic()-st
            state["status"] = "complete_smoke" if args.limit_shots == 3 else "complete"
    except StopRun as exc:
        state["status"], state["stop_reason"] = "stopped_partial", str(exc)
    except Exception as exc:
        state["status"], state["stop_reason"] = "error", f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        checkpoint(state, args, start)
    print(json.dumps(dict(status=state["status"], elapsed_seconds=state["elapsed_seconds"],
                         S={n: r["S"] for n, r in state["results"].items()}), indent=2))
    return 0 if state["status"].startswith("complete") else 2


if __name__ == "__main__":
    raise SystemExit(main())
