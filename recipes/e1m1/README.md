# E1M1 混合239本・seed 17 の再現

Doom E1M1（Hangar）の人間プレイ89本、自走成功50本、自走死亡100本を映像学習に使い、成功50本だけをボタンの教師にした実験です。次の3区間で同じモデルを比較できます。

| 出力名 | 初期状態 | 累積更新 | WM学習率 |
|---|---|---:|---:|
| `base` | seed 17 のランダム初期値 | 0 → 25,000 | 0.0005 |
| `control` | base の25,000更新点 | 25,000 → 50,000 | 0.0005 |
| `higher` | 同じ base の25,000更新点 | 25,000 → 50,000 | 0.00075 |

両分岐はエンコーダー、EMA教師、WM、行動ヘッド、AdamWのモーメント、各サンプラーの位置を継承します。`higher` ではWMの学習率を変更し、最良検証WM損失の集計を分岐後から始めます。元実験の25,000更新評価には、その時点の重みを使用しました。

## 1. 別のマシンに環境を作る

元環境はLinux x86-64、Python 3.12.3、RTX 5090、PyTorch 2.11.0+cu128。学習にはBF16対応のCUDA GPUを使います。学習用RGB配列だけで約31 GiBを使うため、動画・チェックポイントを含めて40 GiB以上の空き容量を用意してください。次のコマンドはリポジトリのルートから実行します。

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r recipes/e1m1/requirements.txt -e .
cd recipes/e1m1
# 任意。省略時はこのディレクトリの data/ と runs/ を使用
export E1M1_DATA="$PWD/data" E1M1_RUNS="$PWD/runs"
```

### ゲーム資材

Debian/Ubuntuでは `git`、`g++`、`libsdl2-dev`、`dpkg`、`coreutils` を用意し、次を実行します。shareware版のIWADと公開されたE1M1のマップソースを取得し、固定コミットのノードビルダーでPWADを生成します。両WADのSHA-256を検査します。手元に同じ2ファイルがあれば `$E1M1_DATA/e1m1_assets/{doom1.wad,E1M1.wad}` に置けます。[マップソース](https://github.com/TCRF/old-doom-stuff/tree/bfc76f379e6967ae82007b6e61d256bf520ab6a1/DoomMapsSE) · [ノードビルダー](https://github.com/teefoss/doombsp/tree/01969e4508a40e69b735df3362da308a0abb3d96)

```bash
bash assets.sh
```

## 2. 同じ入力を作る

**元実験と画素まで揃える場合は、人間動画の転送用ファイル `e1m1-human89.tar`（938 MiB）を別途コピーしてください。** アーカイブは89本の `videos/<DSDA ID>.mp4` を含みます。Gitには各動画の出典、リプレイのメンバー名、クリアタイム、動画・配列のチェックサムを収録しています。[入力一覧](reference/manifest.json) · [人間リプレイの出典と自走seed](reference/sources.json)

```bash
# tar が現在の作業ディレクトリにある場合
echo '3a27c4f791a28e33461e3e7c232161f733602f0d73abe77bb41cc96877133fe7  e1m1-human89.tar' | sha256sum -c -
mkdir -p "$E1M1_DATA"
tar -xf e1m1-human89.tar -C "$E1M1_DATA"
python prepare.py --videos "$E1M1_DATA/videos"
python prepare.py --verify
```

`prepare.py` は自走150本を記録されたseedで再走し、デモを1 ticずつ再生してRGBを取り出します。人間動画を35 fpsで読み、両方を192×144にリサイズして192×192へ黒帯を加えます。準備済みデータを転送する場合は `e1m1_mixed_rgb/` 一式を `$E1M1_DATA/` にコピーし、`python prepare.py --verify` から始められます。検証成功の最後の行は `Verified 239 banks; 11,807 train and 530 validation action labels.` です。

人間デモ取得から追試したい場合は `python human_videos.py --dsda /usr/games/dsda-doom` を使えます。元環境のDSDA-Doomは0.27.5で、FFmpegは `imageio-ffmpeg==0.8.1` のLinux同梱版です。`--id 95301` で1本に限定できます。スクリプトはLMPのハッシュ、タイム、撃破数、秘密数を照合し、元MP4と一致したファイルを保存します。検証では同じデモ・同じクリアタイムでも一部フレームに画素差が出ました。元入力の完全再現には上記アーカイブまたは検証済みRGB配列を使ってください。[DSDA](https://dsdarchive.com/) · [描画設定](reference/dsda.cfg)

### 教師と分割

| データ | 本数 | WM・遷移の教師 | ボタン教師 |
|---|---:|:---:|:---:|
| 人間のUV-Maxリプレイ | 89 | ○ | |
| 自走成功 | 50 | ○ | ○ |
| 自走死亡 | 100 | ○ | |

35 tic/秒、操作の間隔は4 tic。各群の入力順で、23秒以上あり目標用リプレイを除いた候補の末尾5本を選び、各末尾約12秒を検証に使います。他の部分は訓練に使います。同じ走行の前半と末尾が別の分割に入るため、検証WM・操作BCEは同じ走行内での時刻分割の成績です。実走評価には別の100 seedを使います。操作教師は訓練11,807時点、検証530時点。ボタン順と各分割位置はmanifestにあります。595回の収集試行のseed・成否も保存しています。[収集試行](reference/attempts.json)

## 3. 学習と評価を実行する

まずGPU上で、最初の実データバッチの3損失を元実装と照合します。学習開始は次の2番目のコマンドです。systemdユーザーサービスと15分タイマーを登録し、`base → 25k実走100本 → control → higher` を順次実行します。ログアウト後・再起動後も監視するには、そのユーザーのsystemdマネージャーが常駐する設定を使ってください。

```bash
python verify_model.py
python watch.py install
cat "$E1M1_RUNS/progress.md"
```

`progress.md` に更新数・学習WM損失・検証WM損失・検証操作BCEとその測定更新を記録します。`worker.log` に学習曲線が残り、`watch.json` に終了状態、ログ末尾、再開判断を保存します。停止は `python watch.py pause`、再開は同じ環境で `python watch.py install`。破損した最新保存点は直前の保存点を検証して復旧します。同じ位置で失敗が続いた場合はログを確認してください。

**元実験の25,000更新点から始める場合**は、空の `E1M1_RUNS` で `python import_base.py /path/to/resume_at_25000.pt` を一度実行してから監視を起動します。必要な元ファイルは327 MiBの再開用保存点で、SHA-256は `f596b7562588e4ea1c4954826c4582d88620226558f04e062ea3958f8a625107`。`policy_exact_25000.pt` は評価用です。

systemdを使わない計算機では、スケジューラ側に同じ順番と15分監視を設定できます。ワーカーの実行コマンドを以下に示します。分岐先は別ディレクトリに保存され、同じコマンドを再実行すると自分の最新保存点から続きます。

```bash
python experiment.py base
python evaluate.py --checkpoint "$E1M1_RUNS/base/exact.pt" --out "$E1M1_RUNS/eval25000"
python experiment.py control
python experiment.py higher
```

`resume.pt` は再開用、`exact.pt` はその区間の最終重み、`best.pt` は検証WM損失最小の重みです。実走評価は同じ開始姿勢、未使用seed 61000〜61099、難度4、上限120秒。訓練の成功軌跡 `train_000` を目標画像とし、ゲーム座標で判定した経路の進行に応じて次の20/40/60/80/100%地点に切り替えます。出力は各走行の操作列・リプレイ・終了理由・到達度、および集計JSONです。

## モデルと更新条件

入力はRGB3チャネル。4枚の履歴から各時刻の空間ViTのCLSを取り、時系列Transformerで512次元の潜在を生成します。WMは幅512・9段。現在潜在と最終段の予測潜在から16次元の潜在行動を作り、8ボタンを予測します。エンコーダー、WM、行動エンコーダー、遷移予測器、行動デコーダーを共同更新します。

- WMは1〜10秒先から各2件、計20件を毎更新サンプリングします。9段の未来潜在はEMA係数0.996のターゲットエンコーダーで作ります。遷移は8 tic先で20件、ボタンBCEは成功動画から128件です。
- AdamWの学習率はエンコーダー `3e-5`、WMは上表の値、行動ヘッド `5e-4`。勾配ノルム上限1。BF16、学習マイクロバッチ32、検証16。履歴内の重複フレームをまとめ、同じ目標の操作をまとめ、学習時の最終空間attentionではCLSクエリのみを計算します。
- WM検証・保存は100更新ごと、操作BCEは100更新目と以降1,000更新ごとです。移植版は歴史的な初期化の乱数消費順と元のバッチ内順序も復元します。[実装](experiment.py) · [汎用高速化API](../../README.md#高速版の共同学習)

## 比較用の元実験結果

| 元実験の重み | 検証WM損失 | 検証操作BCE | 実走クリア |
|---|---:|---:|---:|
| base 25,000 | 0.66967 | 0.90288 | 0/100 |
| control 50,000 | 0.66184 | 0.97240 | — |

25,000更新では100本とも死亡。20%地点に93本、40%地点に16本が到達し、60%地点に到達した走行は0本でした。`learning_curves.csv` は実測ログのスナップショットで、base・controlは終端まで、higherは39,600更新までを収録します。[元実験の学習曲線](reference/learning_curves.csv) · [100本の個別評価結果](reference/eval25000.json)

元動画から生成した239配列すべてのSHA-256、成功50本のラベル、元の初期モデルのバッチ損失を照合しました。元の25,000更新重みでは検証WM損失・検証操作BCEと、100走すべての操作列・結果が移植前と一致しました。移植版のランダム初期化から50,000更新までの独立再学習と異機種GPUでの数値一致は未検証です。元の評価重みがある場合は `python verify_model.py --original-25000 /path/to/policy_exact_25000.pt` で検証損失を比較できます。
