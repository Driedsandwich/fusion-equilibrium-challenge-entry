---
audience: ai
---

# 提出用の道具とClaudeの実行命令

実行担当はすべてAI（Claude）です。作業場所の根元から `.venv/bin/python -B` を使ってください。アップロードは含みません。環境構築、固定版の指定、MASTを含む再学習の1命令はREADME.mdにあります。

全312本の学習は完了しています。`weights/train_meta.json` の実測は101.226秒です。`datasets` 5.1.0は導入済みです。今回の修正では外部取得・遠隔stream・本番全件は実行していません。古い `_work/submission_smoke/verification.json` は当時の履歴として残しています。

## 手元の開発データで2 shotsずつ試す

DIII-Dは手元の訓練データ、MASTは手元の公開テスト入力を使います。同梱の全312本で学習済みの重みを使います。新しい出力先を指定してください。`--stop 1`で区切り、次の実行で先頭を検査してスキップし、2本目まで進めます。

```bash
set -eu
RUN_DIR="_work/claude_check_$(date +%Y%m%d_%H%M%S)"
for spec in diii_d:diii_d_train mast:mast_public_test; do
  machine="${spec%%:*}"
  folder="${spec#*:}"
  for stage in first resume; do
    if [ "$stage" = first ]; then set -- --stop 1; else set --; fi
    .venv/bin/python -B predict.py --config "${machine}_local_check" --machine "$machine" --local-dir "_data/$folder" --limit-shots 2 --weights weights --work-dir "$RUN_DIR/$machine" --no-guard "$@"
  done
  .venv/bin/python -B predict.py --config "${machine}_local_check" --machine "$machine" --local-dir "_data/$folder" --limit-shots 2 --weights weights --work-dir "$RUN_DIR/$machine" --assemble --no-guard --output "$RUN_DIR/${machine}_private_test.npz"
  .venv/bin/python -B validate.py "$RUN_DIR/${machine}_private_test.npz" --config "${machine}_local_check" --machine "$machine" --local-dir "_data/$folder" --limit-shots 2
done
.venv/bin/python -B make_zip.py --diii-d "$RUN_DIR/diii_d_private_test.npz" --mast "$RUN_DIR/mast_private_test.npz" --output "$RUN_DIR/submission.zip" --no-guard
```

遠隔の開発データで最初の2 shotsの推論だけ試す場合は、下の本番変数へ開発用の値を入れ、predictの推論命令に `--stop 2` を追加してください。この段階では全streamの総数が未確定なので、組み立てはしません。`--start 2`で再開して最後まで進めれば組み立てられます。

## 本番

発表されたrepo・split・両装置のconfig・固定データコミットSHAを設定してください。以下の仮置き値のまま実行しないでください。開発段階ではrepo=`Sophelio/fusion-equilibrium-challenge`、split=`public_test`、config=`diii_d_public_test` と `mast_public_test` です。

```bash
export DATASET_REPO='REPLACE_WITH_ANNOUNCED_REPO'
export DATASET_REVISION='REPLACE_WITH_FIXED_DATASET_COMMIT_SHA'
export DATASET_SPLIT='REPLACE_WITH_ANNOUNCED_SPLIT'
export D3_CONFIG='REPLACE_WITH_ANNOUNCED_DIII_D_CONFIG'
export MAST_CONFIG='REPLACE_WITH_ANNOUNCED_MAST_CONFIG'
export RUN_DIR="_work/final_$(date +%Y%m%d_%H%M%S)"
export WEIGHTS='weights'
set -eu
for machine in diii_d mast; do
  config="$D3_CONFIG"
  if [ "$machine" = mast ]; then config="$MAST_CONFIG"; fi
  .venv/bin/python -B predict.py --repo "$DATASET_REPO" --revision "$DATASET_REVISION" --config "$config" --split "$DATASET_SPLIT" --machine "$machine" --weights "$WEIGHTS" --work-dir "$RUN_DIR/$machine"
  .venv/bin/python -B predict.py --repo "$DATASET_REPO" --revision "$DATASET_REVISION" --config "$config" --split "$DATASET_SPLIT" --machine "$machine" --weights "$WEIGHTS" --work-dir "$RUN_DIR/$machine" --assemble --output "$RUN_DIR/${machine}_private_test.npz"
  .venv/bin/python -B validate.py "$RUN_DIR/${machine}_private_test.npz" --repo "$DATASET_REPO" --revision "$DATASET_REVISION" --config "$config" --split "$DATASET_SPLIT" --machine "$machine"
done
.venv/bin/python -B make_zip.py --diii-d "$RUN_DIR/diii_d_private_test.npz" --mast "$RUN_DIR/mast_private_test.npz" --output "$RUN_DIR/submission.zip"
```

途中停止後は同じ引数で再実行してください。再開・組み立ての両方でrepo/config/split/revision・重み・MAST校正・推論コード・ローカル入力のSHA256を照合します。各chunkのsource・T・実行条件も照合します。組み立て時にも入力・重みの引数を省略しないでください。旧形式manifestや不一致の作業先は、新しい作業先で最初から作り直してください。既存成果物や中断したpartialは削除しません。

MASTはDIII-Dから得たj1のα_Fとκ・q95・βNを使い、原点c=0、ソレノイド1倍とします。MASTの平衡真値は読みません。予測は全efit_timesを保存します。zipの直下は正規名のnpz 2本のみです。
