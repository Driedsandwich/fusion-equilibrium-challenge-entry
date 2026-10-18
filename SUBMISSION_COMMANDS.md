---
audience: ai
---

# 提出用の道具とClaudeの実行命令

実装と手元の2本ずつの動作確認は完了しました。全312本の学習、Hugging Faceからの取得、実際の流し読みは未実行です。`datasets`はこの環境には入っていません。Claudeが導入してから外部データの経路を確認してください。以下の実行担当はすべてAI（Claude）です。アップロードは含みません。

作業場所は `.` です。Pythonは `.venv/bin/python -B` を使います。既存の重み・完成npz・zipを上書きしないので、再実行時は新しい出力先を指定してください。

## 取得と全312本の学習

```bash
# 既にある312本を使う場合、fetch_train.pyの実行は不要です。
.venv/bin/python -B fetch_train.py
.venv/bin/python -B train.py
```

`train_shots.txt`は現在の312本のファイル名です。全件学習に保留はありません。版dと同じく各shotからPCA用8フレーム、回帰用最大64フレームを選びます。121特徴、PCA20成分、成分ごとのHistGradientBoosting最大60反復、スカラー各100反復、種42です。出力は `weights/model.joblib` と `weights/train_meta.json` です。

## 手元の2本ずつの再現試験

初回の試験結果は `_work/submission_smoke/verification.json`、提出形式のzipは `_work/submission_smoke/submission.zip` にあります。DIII-Dの全418フレーム、MASTの全87フレームを保存し、再開時の既存shotのスキップも確認しました。型違い、非有限値、欠けたキー、余分なキー、欠けたフレーム、不完全な組み立ては拒否しました。性能評価は行っていません。

以下は新しい作業先を使う再現命令です。`--stop`は指定した番号を含みません。

```bash
set -e
.venv/bin/python -B train.py --limit-shots 2 --no-guard --weights _work/local_check/weights
for spec in diii_d:diii_d_train mast:mast_public_test; do
  machine="${spec%%:*}"
  folder="${spec#*:}"
  for stop in 1 2; do
    .venv/bin/python -B predict.py --config "${machine}_local_check" --machine "$machine" --local-dir "_data/$folder" --limit-shots 2 --weights _work/local_check/weights --work-dir "_work/local_check/$machine" --start 0 --stop "$stop" --no-guard
  done
  .venv/bin/python -B predict.py --config "${machine}_local_check" --machine "$machine" --work-dir "_work/local_check/$machine" --assemble --no-guard --output "_work/local_check/${machine}_private_test.npz"
  .venv/bin/python -B validate.py "_work/local_check/${machine}_private_test.npz" --config "${machine}_local_check" --machine "$machine" --local-dir "_data/$folder" --limit-shots 2
done
.venv/bin/python -B make_zip.py --diii-d _work/local_check/diii_d_private_test.npz --mast _work/local_check/mast_private_test.npz --output _work/local_check/submission.zip
```

## 開発データでの試しと本番

まず開発データでは次の変数を使います。全312本で学習済みの `weights/` が必要です。

```bash
DATASET_REPO='Sophelio/fusion-equilibrium-challenge'
DATASET_SPLIT='public_test'
D3_CONFIG='diii_d_public_test'
MAST_CONFIG='mast_public_test'
RUN_DIR='_work/development'
```

本番では上の5変数を、発表されたrepo・split・DIII-D設定名・MAST設定名、および別の作業先 `_work/final` に設定し直します。予定のrepo名を確定値として使わないでください。設定名が装置名で始まらなくても `--machine` で対応します。どちらの場合も以下を実行します。

```bash
set -e
for machine in diii_d mast; do
  config="$D3_CONFIG"
  if [ "$machine" = mast ]; then config="$MAST_CONFIG"; fi
  .venv/bin/python -B predict.py --repo "$DATASET_REPO" --config "$config" --split "$DATASET_SPLIT" --machine "$machine" --work-dir "$RUN_DIR/$machine" --stop 2
  .venv/bin/python -B predict.py --repo "$DATASET_REPO" --config "$config" --split "$DATASET_SPLIT" --machine "$machine" --work-dir "$RUN_DIR/$machine" --start 2
  .venv/bin/python -B predict.py --config "$config" --machine "$machine" --work-dir "$RUN_DIR/$machine" --assemble --output "$RUN_DIR/${machine}_private_test.npz"
  .venv/bin/python -B validate.py "$RUN_DIR/${machine}_private_test.npz" --repo "$DATASET_REPO" --config "$config" --split "$DATASET_SPLIT" --machine "$machine"
done
.venv/bin/python -B make_zip.py --diii-d "$RUN_DIR/diii_d_private_test.npz" --mast "$RUN_DIR/mast_private_test.npz" --output "$RUN_DIR/submission.zip"
```

全shotを一度で推論する場合は、最初の2回のpredict命令を `--start` と `--stop` なしの1回にまとめられます。途中停止後は同じ引数で再実行すると保存済みshotを検査してスキップします。元のstreamを先頭から順に走査するので、後半からの再開でも入力の読込は発生します。最後まで走査してshot総数が確定し、全chunkが揃うまでは組み立てません。手元の試験の番号はファイル名の昇順です。本番番号はdatasetsのstream順を0から数えます。

MASTは既存 `mast/simple.py` を再利用し、j1のα_F、原点c=0、ソレノイド1倍を使います。κ・q95・βNは既存のDIII-D中央値です。MASTのEFIT真値は読みません。推論・組み立てはshot単位で、npzを全体展開しません。各shotの前に既存のスワップ読み取りとdfを使い、空き300 MB／12 GiB未満または取得不能なら停止します。`--no-guard`は手元試験用です。提出zipは内部のnpzを再圧縮せず、根元に `diii_d_private_test.npz` と `mast_private_test.npz` の2本だけを置きます。
